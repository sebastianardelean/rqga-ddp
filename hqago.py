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
from rqga import rqga, oracle,grover, to_binary, bin_to_dec,bin_to_dec_msb





def recreate_individual(opt_individual:[], individual_binary:[], no_of_qubits_individual:int) -> []:
    solution = [(0,False) for _ in range(0,no_of_qubits_individual)]

    for gene in opt_individual:
        position=abs(gene)-1
        if gene>0:
            new_value = (1,True)
            solution[position]=new_value
        else:
            new_value = (0, True)
            solution[position]=new_value
    for gene in individual_binary:
        for (value,marked),idx in enumerate(solution):
            if marked == False:
                solution[idx] = (gene, True)
                break
    return list(map(lambda x: x[0], solution))
# In[12]:



def get_optimized_individuals(opt_individual:[], individual_binary:[]):
    new_individuals = []
    for individual in individual_binary:
        new_individuals.append((individual, False))
    for gene in opt_individual:
        position = abs(gene) - 1
        new_value = (1,True) if gene > position else (0,False)
        new_individuals[position] = new_value
    return new_individuals




def ufit_optimized(population_size:int,no_of_qubits_individual:int, no_of_qubits_fitness:int,fitness_function:callable, opt_individual) -> Instruction:
    # define quantum registers
    ind_qreg = QuantumRegister(no_of_qubits_individual, "ind_qreg")
    fit_qreg = QuantumRegister(no_of_qubits_fitness + 1, "fit_qreg")

    # create the U_fit subcircuit
    qc = QuantumCircuit(ind_qreg, fit_qreg, name="U$_fit$")


    for i in range(population_size):
        controls = []
        # prepare individual
        individual_binary = to_binary(i, no_of_qubits_individual, True)

        optimized_individual = get_optimized_individuals(opt_individual, individual_binary)

        for (value, marked), index in enumerate(optimized_individual):
            if value == 0 and marked == False:
                qc.x(ind_qreg[index])
                controls.append(ind_qreg[index])

        mcx = MCXGate(len(controls))  # reusable gate

        # compute fitness
        valid, fitness_value = fitness_function(individual_binary)
        fitness_value_binary = to_binary(
            fitness_value, no_of_qubits_fitness, True
        )

        # write fitness value
        for k in range(no_of_qubits_fitness):
            if fitness_value_binary[k] == 1:
                qc.append(mcx, controls + [fit_qreg[k]])

        # mark validity bit
        if valid:
            qc.append(mcx, controls + [fit_qreg[no_of_qubits_fitness]])


        # uncompute individual
        for (value, marked), index in enumerate(optimized_individual):
            if value == 0 and marked == False:
                qc.x(ind_qreg[index])

        qc.barrier()

    return qc.to_instruction()


def hqago_run(
        no_of_qubits_individual: int,
        no_of_qubits_fitness: int,
        no_of_generations: int,
        fitness_function: callable,
        initial_max_value: int,
        individual_optimization: [],
)->list[(int,int)]:
    rqga_results = []
    max_value = initial_max_value
    new_fitness_value = 0
    generation = 0
    no_of_grover_iterations = round(math.pi/4 * math.sqrt(2**no_of_qubits_fitness) - 0.5)
    population_size = 2**no_of_qubits_individual
    while new_fitness_value != max_value and generation < no_of_generations:
        # while generation < 1:
        print("Creating the circuit!")
        ind_qreg = QuantumRegister(no_of_qubits_individual, "i")
        fit_qreg = QuantumRegister(no_of_qubits_fitness + 1, "f")
        carry_in = QuantumRegister(1, "cin")
        carry_out = QuantumRegister(1, "cout")
        carry_help = QuantumRegister(1, "help")
        ws = QuantumRegister(1, "ws")
        max_qreg = QuantumRegister(no_of_qubits_fitness, "max")
        creg = ClassicalRegister(no_of_qubits_fitness, "m")
        qc = QuantumCircuit(ind_qreg, fit_qreg, carry_in, max_qreg, ws, creg)

        print("Creating the superposition")
        qc.h(ind_qreg)
        qc.x(ws)
        qc.h(ws)

        print("Applying ufit")

        ufit_gate = ufit_optimized(population_size,no_of_qubits_individual, no_of_qubits_fitness,fitness_function, individual_optimization)

        qc.append(ufit_gate, ind_qreg[:] + fit_qreg[:])
        max_value = new_fitness_value
        max_value_bin = to_binary(max_value, no_of_qubits_fitness, True)

        print("Encoding max value")
        # Encode Max value into max_reg
        for i in range(0, no_of_qubits_fitness):
            if max_value_bin[i] == 1:
                qc.x(max_qreg[i])

        print("Creating the oracle gate")
        oracle_gate = oracle(max_value_bin,no_of_qubits_fitness, no_of_qubits_individual)
        print("Creating grover gate")
        grover_gate = grover(no_of_qubits_fitness)

        # for _ in range(0, no_of_grover_iterations):
        for _ in range(0, 1):
            print("Apply grover")
            qc.append(oracle_gate, fit_qreg[:] + max_qreg[:] + carry_in[:] + carry_out[:] + carry_help[:] + ws[:])
            qc.append(grover_gate, fit_qreg[:])

        qc.barrier()

        qc.measure(fit_qreg,creg)
        simulator = AerSimulator(method="statevector", device="GPU")

        qc = transpile(qc, simulator)
        results = simulator.run(qc).result()
        counts = results.get_counts()
        max_item = max(counts.items(), key=operator.itemgetter(1))
        print(f"Counts are: {counts} and max is:{max_item}")
        # recreated_individual = recreate_individual(individual_optimization, max_item,no_of_qubits_individual)

        new_fitness_value = bin_to_dec_msb(max_item[0])
        print(
            f"Found solution with fitness {new_fitness_value} in generation {generation} with {no_of_grover_iterations} grover iterations")
        rqga_results.append((generation, new_fitness_value))

        generation += 1
        # reset max value qreg
        for i in range(0, no_of_qubits_fitness):
            if max_value_bin[i] == 1:
                qc.x(max_qreg[i])
    return rqga_results


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
        opt_population_size:int,
        opt_selection_method:str,
        opt_num_parents_to_keep:int,
        opt_crossover_rate:float,
        opt_crossover_method:str,
        opt_mutation_rate:float,
        opt_mutation_method:str,
        opt_solution_per_populations:int,
        opt_fitness_function:callable
) -> list[(int, int)]:

    result = []
    if use_optimization == True:
        if opt_gene_length == no_of_qubits_individual:
            # We have classic ga, so just run an return
            ga = pygad.GA(
                num_generations=opt_number_of_generations,
                num_parents_mating=opt_num_parents_mating,
                gene_type=int,
                fitness_func=fitness_function,
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
            result.append((generation, fitness))
            return result
        elif opt_gene_length == 0:
            # the classic RQGA
            return rqga(
                no_of_qubits_individual,
                no_of_qubits_fitness,
                no_of_generations,
                fitness_function,
                initial_max_value
            )
        else:
            opt_gene_low = -(no_of_qubits_individual + 1)
            opt_gene_high = no_of_qubits_individual + 1
            ga = pygad.GA(
                num_generations=opt_number_of_generations,
                num_parents_mating=opt_num_parents_mating,
                gene_type=int,
                fitness_func=opt_fitness_function,
                sol_per_pop=opt_solution_per_populations,
                num_genes=opt_gene_length,
                gene_space=[0, 1],
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
            return hqago_run(
                no_of_qubits_individual,
                no_of_qubits_fitness,
                no_of_generations,
                fitness_function,
                initial_max_value,
                solution.tolist()
            )



