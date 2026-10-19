import pygad
from qiskit import QuantumCircuit, transpile, QuantumRegister,ClassicalRegister
from qiskit.circuit import Instruction
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCXGate
from qiskit.circuit.library import grover_operator
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.circuit.library import FullAdderGate
import numpy as np
import math
import operator
from rqga import (
    rqga,
    oracle,
    diffuser,
    to_binary,
    bin_to_dec,
    bin_to_dec_msb,
    select_simulator,
)


# HQAGO chromosome encoding (Ardelean & Udrescu, 2024, Fig. 3): the absolute
# value of a gene is the 1-based position of a fixed qubit in the individuals'
# register and the sign gives the value that qubit is fixed to (+v -> 1,
# -v -> 0). A gene equal to 0 leaves the corresponding position quantum.
# Because duplicate genes are allowed, the last gene for a position wins.
def fixed_gene_values(opt_individual:[], no_of_qubits_individual:int) -> []:
    fixed = [None] * no_of_qubits_individual
    for gene in opt_individual:
        gene = int(gene)
        if gene == 0:
            continue
        position = abs(gene) - 1
        if not 0 <= position < no_of_qubits_individual:
            raise ValueError(
                f"gene {gene} points outside the individuals' register "
                f"(1..{no_of_qubits_individual})"
            )
        fixed[position] = 1 if gene > 0 else 0
    return fixed


def recreate_individual(opt_individual:[], individual_binary:[], no_of_qubits_individual:int) -> []:
    # Apply the classical GA genes to a (partial) free-qubit assignment. The
    # fixed positions take the gene values; the remaining positions are filled,
    # in increasing position order, with the values of individual_binary.
    solution = [(0, False) for _ in range(0,no_of_qubits_individual)]

    for gene in opt_individual:
        gene = int(gene)
        if gene == 0:
            continue
        position = abs(gene) - 1
        solution[position] = (1 if gene > 0 else 0, True)
    for gene in individual_binary:
        for idx, (value, marked) in enumerate(solution):
            if marked == False:
                solution[idx] = (int(gene), True)
                break
    return list(map(lambda x: x[0], solution))


def ufit_optimized(population_size:int,no_of_qubits_individual:int, no_of_qubits_fitness:int,fitness_function:callable, opt_individual) -> Instruction:
    # U_fit over the reduced search space S_k (|S_k| = 2**(n-k)). Only the free
    # (non-fixed) qubits are enumerated; the fixed qubits keep the classical
    # values selected by the GA, and the fitness is evaluated on the complete
    # chromosome (fixed bits + current free-qubit assignment).
    fixed = fixed_gene_values(opt_individual, no_of_qubits_individual)
    free_positions = [j for j in range(no_of_qubits_individual) if fixed[j] is None]
    reduced_population_size = 2 ** len(free_positions)
    if population_size != reduced_population_size:
        raise ValueError(
            f"population_size={population_size} does not match the reduced "
            f"search space 2**{len(free_positions)}={reduced_population_size}"
        )

    # define quantum registers
    # The fitness value is encoded on no_of_qubits_fitness + 1 qubits
    # (two's complement, sign bit included) plus one validity qubit.
    value_bits = no_of_qubits_fitness + 1
    ind_qreg = QuantumRegister(no_of_qubits_individual, "ind_qreg")
    fit_qreg = QuantumRegister(value_bits + 1, "fit_qreg")

    # create the U_fit subcircuit
    qc = QuantumCircuit(ind_qreg, fit_qreg, name="U_fit")

    # Only the free qubits are control qubits: the fixed ones are prepared by
    # the caller in a computational basis state, so they carry no information
    # for the search and must not appear in the controlled gates.
    controls = [ind_qreg[j] for j in free_positions]
    mcx = MCXGate(len(controls)) if len(controls) > 0 else None

    for i in range(reduced_population_size):
        # build the complete individual for the i-th reduced-space element
        free_bits = to_binary(i, len(free_positions), True)
        individual = recreate_individual(
            opt_individual, free_bits, no_of_qubits_individual
        )

        # prepare the free-qubit basis state of this individual
        for position in free_positions:
            if individual[position] == 0:
                qc.x(ind_qreg[position])

        # compute fitness
        valid, fitness_value = fitness_function(individual)
        fitness_value_binary = to_binary(fitness_value, value_bits, True)

        # write fitness value (sign bit included)
        for k in range(value_bits):
            if fitness_value_binary[k] == 1:
                if mcx is not None:
                    qc.append(mcx, controls + [fit_qreg[k]])
                else:
                    qc.x(fit_qreg[k])

        # mark validity bit
        if valid:
            if mcx is not None:
                qc.append(mcx, controls + [fit_qreg[value_bits]])
            else:
                qc.x(fit_qreg[value_bits])

        # uncompute individual
        for position in free_positions:
            if individual[position] == 0:
                qc.x(ind_qreg[position])

        qc.barrier()

    return qc.to_instruction()


def hqago_run(
        no_of_qubits_individual: int,
        no_of_qubits_fitness: int,
        no_of_generations: int,
        fitness_function: callable,
        initial_max_value: int,
        individual_optimization: [],
        grover_iterations: int = None,
)->list[(int,int)]:
    fixed = fixed_gene_values(individual_optimization, no_of_qubits_individual)
    free_positions = [
        j for j in range(no_of_qubits_individual) if fixed[j] is None
    ]
    no_of_fixed_qubits = no_of_qubits_individual - len(free_positions)
    population_size = 2 ** len(free_positions)

    if not free_positions:
        individual = recreate_individual(
            individual_optimization, [], no_of_qubits_individual
        )
        valid, value = fitness_function(individual)
        if valid and value > initial_max_value:
            return [(0, value)]
        return []

    print(
        f"HQAGO: {no_of_fixed_qubits} fixed qubit(s) "
        f"{[(j, fixed[j]) for j in range(no_of_qubits_individual) if fixed[j] is not None]}, "
        f"{len(free_positions)} free qubit(s), reduced population {population_size}"
    )

    rqga_results = []
    max_value = initial_max_value
    generation = 0
    simulator = select_simulator()

    while generation < no_of_generations:
        print("Creating the circuit!")
        ind_qreg = QuantumRegister(no_of_qubits_individual, "i")
        fit_qreg = QuantumRegister(no_of_qubits_fitness + 2, "f")
        carry_in = QuantumRegister(1, "cin")
        carry_out = QuantumRegister(1, "cout")
        sign_copy = QuantumRegister(1, "signcopy")
        ws = QuantumRegister(1, "ws")
        max_qreg = QuantumRegister(no_of_qubits_fitness + 1, "max")
        creg = ClassicalRegister(len(free_positions), "m")
        qc = QuantumCircuit(ind_qreg, fit_qreg, carry_in, carry_out, sign_copy, max_qreg, ws, creg)

        marked = 0
        for i in range(population_size):
            free_bits = to_binary(i, len(free_positions), True)
            individual = recreate_individual(
                individual_optimization, free_bits, no_of_qubits_individual
            )
            valid, value = fitness_function(individual)
            if valid and value > max_value:
                marked += 1
        if marked == 0:
            break
        if grover_iterations is None:
            iterations = max(
                0,
                int(round(math.pi / 4 * math.sqrt(population_size / marked) - 0.5)),
            )
        else:
            iterations = grover_iterations
        print(f"Marked individuals: {marked}/{population_size}, using {iterations} grover iteration(s)")

        # Superposition over the reduced search space only: free qubits are in
        # superposition, fixed qubits are prepared as classical values.
        print("Creating the reduced superposition")
        for position in range(no_of_qubits_individual):
            if fixed[position] is None:
                qc.h(ind_qreg[position])
            elif fixed[position] == 1:
                qc.x(ind_qreg[position])
        qc.x(ws)
        qc.h(ws)

        diffuser_qubits = [ind_qreg[j] for j in free_positions]

        if iterations > 0:
            print("Applying ufit")
            ufit_gate = ufit_optimized(population_size,no_of_qubits_individual, no_of_qubits_fitness,fitness_function, individual_optimization)
            max_value_bin = to_binary(max_value, no_of_qubits_fitness + 1, True)

            print("Encoding max value")
            for i in range(0, no_of_qubits_fitness + 1):
                if max_value_bin[i] == 1:
                    qc.x(max_qreg[i])

            print("Creating the oracle gate")
            oracle_gate = oracle(max_value_bin,no_of_qubits_fitness, no_of_qubits_individual)
            print("Creating grover gate")
            diffuser_gate = diffuser(len(free_positions))

            for i in range(iterations):
                print(f"Starting iteration {i}")
                print("\tApply ufit")
                qc.append(ufit_gate, ind_qreg[:] + fit_qreg[:])
                print("\tApply Grover's oracle")
                qc.append(oracle_gate, fit_qreg[:] + max_qreg[:] + carry_in[:] + carry_out[:] + sign_copy[:] + ws[:])
                print("\tUncompute ufit")
                qc.append(ufit_gate.inverse(), ind_qreg[:] + fit_qreg[:])
                print("\tApply Grover's diffuser")
                qc.append(diffuser_gate, diffuser_qubits)

        qc.barrier()
        print("Measure the circuit")
        qc.measure(diffuser_qubits, creg)
        print("Transpile")
        qc = transpile(qc, simulator,
                   optimization_level=0)
        print("Qubits:", qc.num_qubits)
        print("Depth:", qc.depth())
        print("Instructions:", len(qc.data))
        print("Largest operation counts:", qc.count_ops())
        for attempt in range(0, 8):
            print("Run simulation")
            results = simulator.run(qc).result()
            counts = results.get_counts()
            max_item = max(counts.items(), key=operator.itemgetter(1))
            print(f"Counts are: {counts} and max is:{max_item}")
            free_bits = to_binary(int(max_item[0], 2), len(free_positions), True)
            individual = recreate_individual(
                individual_optimization, free_bits, no_of_qubits_individual
            )
            valid, new_fitness_value = fitness_function(individual)
            print(
                f"Found solution with fitness {new_fitness_value} in generation {generation} with {iterations} grover iterations")
            if valid and new_fitness_value > max_value:
                break
            if attempt < 7:
                print("No improvement measured, retrying the measurement")
        else:
            break
        max_value = new_fitness_value
        rqga_results.append((generation, max_value))
        generation += 1
    return rqga_results


def pygad_fitness_function(fitness_function:callable) -> callable:
    # pygad evaluates fitness_func(ga_instance, solution, solution_idx) and
    # expects a single number, while the research fitness functions used by
    # RQGA return (valid, fitness_value). Adapt the two conventions here.
    def fitness_func(ga_instance, solution, solution_idx):
        result = fitness_function(solution)
        if isinstance(result, (tuple, list, np.ndarray)):
            valid = bool(result[0])
            value = float(result[1]) if len(result) > 1 else 0.0
            # pygad maximizes; invalid individuals must never outrank valid ones
            return value if valid else 0.0
        return float(result)
    return fitness_func


def hqago(
        no_of_qubits_individual:int,
        no_of_qubits_fitness:int,
        no_of_generations:int,
        fitness_function:callable,
        initial_max_value:int,
        use_optimization:bool,
        opt_number_of_generations:int,
        opt_gene_length:int,
        opt_num_parents_mating:int,
        opt_selection_method:str,
        opt_num_parents_to_keep:int,
        opt_crossover_rate:float,
        opt_crossover_method:str,
        opt_mutation_rate:float,
        opt_mutation_method:str,
        opt_solution_per_populations:int,
        opt_fitness_function:callable,
        grover_iterations:int = None,
) -> list[(int, int)]:

    result = []
    if use_optimization == False:
        # no classical optimization -> the plain RQGA
        return rqga(
            no_of_qubits_individual,
            no_of_qubits_fitness,
            no_of_generations,
            fitness_function,
            initial_max_value,
            no_of_grover_iterations=(
                1 if grover_iterations is None else grover_iterations
            ),
        )
    if opt_gene_length == no_of_qubits_individual:
        # We have classic ga, so just run an return
        ga = pygad.GA(
            num_generations=opt_number_of_generations,
            num_parents_mating=opt_num_parents_mating,
            gene_type=int,
            fitness_func=pygad_fitness_function(fitness_function),
            sol_per_pop=opt_solution_per_populations,
            num_genes=opt_gene_length,
            gene_space=[0,1],
            parent_selection_type=opt_selection_method,
            crossover_type=opt_crossover_method,
            crossover_probability=opt_crossover_rate,
            mutation_type=opt_mutation_method,
            mutation_probability=opt_mutation_rate,
            keep_parents=opt_num_parents_to_keep,
            allow_duplicate_genes=True,
        )
        ga.run()
        solution, fitness, idx = ga.best_solution()
        generation = ga.generations_completed
        result.append((generation, int(fitness)))
        return result
    elif opt_gene_length == 0:
        # the classic RQGA
        return rqga(
            no_of_qubits_individual,
            no_of_qubits_fitness,
            no_of_generations,
            fitness_function,
            initial_max_value,
            no_of_grover_iterations=(
                1 if grover_iterations is None else grover_iterations
            ),
        )
    else:
        # Hybrid mode: the classical GA fixes qubit positions (+/-position
        # genes, 0 = untouched) in order to shrink the quantum search space;
        # the quantum part then searches only the remaining free qubits.
        gene_space = (
            list(range(-no_of_qubits_individual, 0))
            + [0]
            + list(range(1, no_of_qubits_individual + 1))
        )
        ga = pygad.GA(
            num_generations=opt_number_of_generations,
            num_parents_mating=opt_num_parents_mating,
            gene_type=int,
            fitness_func=pygad_fitness_function(opt_fitness_function),
            sol_per_pop=opt_solution_per_populations,
            num_genes=opt_gene_length,
            gene_space=gene_space,
            parent_selection_type=opt_selection_method,
            crossover_type=opt_crossover_method,
            crossover_probability=opt_crossover_rate,
            mutation_type=opt_mutation_method,
            mutation_probability=opt_mutation_rate,
            keep_parents=opt_num_parents_to_keep,
            allow_duplicate_genes=True,
        )
        ga.run()
        solution, fitness, idx = ga.best_solution()
        generation = ga.generations_completed
        fixed_genes = solution.tolist()
        print(
            f"Classical GA fixed {sum(1 for gene in fixed_genes if gene != 0)} "
            f"position(s) after {generation} generation(s): {fixed_genes} "
            f"(GA fitness {fitness})"
        )
        return hqago_run(
            no_of_qubits_individual,
            no_of_qubits_fitness,
            no_of_generations,
            fitness_function,
            initial_max_value,
            fixed_genes,
            grover_iterations=grover_iterations
        )
