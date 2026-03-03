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

def to_binary(value, number_of_bits, lsb=False)->np.ndarray:
    if lsb == True:
        return np.flip(np.array(list(np.binary_repr(value, number_of_bits)), dtype=int))
    return np.array(list(np.binary_repr(value, number_of_bits)), dtype=int)

def bin_to_dec(bit_str) -> int:
    bits = np.array(list(bit_str), dtype=int)  # convert to array
    decimal = np.dot(bits, 2 ** np.arange(bits.size))  # unsigned value
    decimal = decimal - (1 << bits.size) * (bits[-1])  # apply 2's complement if MSB=1
    return int(decimal)

def bin_to_dec_msb(bit_str) -> int:
    bits = np.array(list(bit_str), dtype=int)           # covert to array
    decimal = np.dot(bits[::-1], 2 ** np.arange(bits.size))  # treat LSB at index 0
    decimal = decimal - (1 << bits.size) * bits[0]     # subtract 2^n if MSB=1
    return int(decimal)


def ufit(population_size:int,no_of_qubits_individual:int, no_of_qubits_fitness:int,fitness_function:callable) -> Instruction:
    # define quantum registers
    ind_qreg = QuantumRegister(no_of_qubits_individual, "ind_qreg")
    fit_qreg = QuantumRegister(no_of_qubits_fitness + 1, "fit_qreg")

    # create the U_fit subcircuit
    qc = QuantumCircuit(ind_qreg, fit_qreg, name="U$_fit$")

    controls = [ind_qreg[j] for j in range(no_of_qubits_individual)]
    mcx = MCXGate(len(controls))  # reusable gate

    for i in range(population_size):
        # prepare individual
        individual_binary = to_binary(i, no_of_qubits_individual, True)
        for k in range(no_of_qubits_individual):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])


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
        for k in range(no_of_qubits_individual):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])

        qc.barrier()

    return qc.to_instruction()


def oracle(max_value, no_of_qubits_fitness:int, no_of_qubits_individual:int) -> Instruction:
    # define and initialize fitness quantum register
    fit_reg = QuantumRegister(no_of_qubits_fitness + 1, "fqreg")
    # define and initialize max quantum register
    max_reg = QuantumRegister(no_of_qubits_fitness, "max")
    # define and initialize carry quantum register
    carry_in = QuantumRegister(1, "cin")
    carry_out = QuantumRegister(1, "cout")
    carry_help = QuantumRegister(1, "help")
    # define and initialize oracle workspace quantum register
    oracle = QuantumRegister(1, "oqreg")

    subtractor = CDKMRippleCarryAdder(no_of_qubits_fitness , kind='full')
    adder = CDKMRippleCarryAdder(no_of_qubits_fitness , kind='fixed')
    oracle_circ = QuantumCircuit(fit_reg, max_reg, carry_in, oracle, name="O")
    # Encode for subtraction

    oracle_circ.x(carry_in[0])
    oracle_circ.x(max_reg)

    oracle_circ.append(subtractor, [carry_in[0]] + max_reg[:] + fit_reg[:-1] + [carry_out[0]])



    # flip ancilla if fitreg = 0. After subtraction, if max is the actual fitness value, then the result in fitreg = 0 -> solution
    # keep in mind that the valid is 1 if the individual is valid.
    # Therefore, if fitness is the target, all the bits should be 0, except valid.
    oracle_circ.x(fit_reg)
    oracle_circ.x(fit_reg[no_of_qubits_fitness])
    controls = [fit_reg[i] for i in range(no_of_qubits_fitness + 1)]
    mcx = MCXGate(len(controls))  # reusable gate
    oracle_circ.append(mcx, controls + [oracle[0]])
    oracle_circ.x(fit_reg[no_of_qubits_fitness])
    oracle_circ.x(fit_reg)




    oracle_circ.append(adder, max_reg[:] + fit_reg[:-1] + [carry_help[0]])

    oracle_circ.cx(max_reg[0], fit_reg[0])




    return oracle_circ.to_instruction()


def grover(no_of_qubits_fitness:int) -> Instruction:
    fit_qreg = QuantumRegister(no_of_qubits_fitness + 1, "f")

    # create grover diffuser subcircuit
    grover_circ = QuantumCircuit(fit_qreg, name="U$_s$")

    grover_circ.h(fit_qreg)
    grover_circ.x(fit_qreg)

    grover_circ.h(fit_qreg[no_of_qubits_fitness])

    controls = [fit_qreg[i] for i in range(no_of_qubits_fitness)]
    mcx = MCXGate(len(controls))  # reusable gate
    grover_circ.append(mcx, controls + [fit_qreg[no_of_qubits_fitness]])

    grover_circ.h(fit_qreg[no_of_qubits_fitness])

    grover_circ.x(fit_qreg)
    grover_circ.h(fit_qreg)
    # print(f"Grover:\n{grover_circ.draw()}")

    return grover_circ.to_instruction()


def rqga(
        no_of_qubits_individual:int,
        no_of_qubits_fitness:int,
        no_of_generations:int,
        fitness_function:callable,
        initial_max_value:int,
        no_of_fixed_genes: int = 0


) -> list[(int, int)]:
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
        ufit_gate = ufit(population_size,no_of_qubits_individual, no_of_qubits_fitness,fitness_function)
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