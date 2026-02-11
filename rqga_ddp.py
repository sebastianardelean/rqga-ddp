import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import os
import math
import operator
# import matplotlib as mpl
import matplotlib.pyplot as plt


# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "plots")
os.makedirs(IMAGES_PATH, exist_ok=True)


def get_path(fig_name, fig_extension="pdf"):
    path = os.path.join(IMAGES_PATH, fig_name + "." + fig_extension)
    return path

from qiskit import QuantumCircuit, transpile, QuantumRegister,ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCXGate
from qiskit.circuit.library import grover_operator
from qiskit.circuit.library import CDKMRippleCarryAdder

from collections import namedtuple
ItemConfiguration = namedtuple('Item', 'm,v')

NO_OF_QUBITS_FITNESS = 8
NO_OF_QUBITS_INDIVIDUAL = 4
POPULATION_SIZE = 2**NO_OF_QUBITS_INDIVIDUAL
NO_OF_MAX_GROVER_ITERATIONS = int(math.sqrt(2**NO_OF_QUBITS_INDIVIDUAL-1))
NO_OF_GENERATIONS = 50

MAX_ALLOWED_MASS = 10
TOTAL_VALUE = 22
items = [ItemConfiguration(3, 3),ItemConfiguration(2, 5),ItemConfiguration(4, 10),ItemConfiguration(7, 4)]
#best result is [1, 1, 1, 0]	9	18	True


rng = np.random.default_rng()


def to_binary(value, number_of_bits, lsb=False):
    """
    Function return two's complement representation
    :param value: value in decimal representation
    :param number_of_bits: number of bits used for representation
    :returns: np.array that represents the binary representation
    >>> to_binary(10,4)
    array([1, 0, 1, 0])
    >>> to_binary(10,4,True)
    array([0, 1, 0, 1])
    """
    if lsb == True:
        return np.flip(np.array(list(np.binary_repr(value, number_of_bits)), dtype=int))
    return np.array(list(np.binary_repr(value, number_of_bits)), dtype=int)

def get_random_value(min_v, max_v):
    return rng.integers(min_v,max_v)

def fitness_function(mass,value):
    return int(value - (TOTAL_VALUE+1)*divmod(mass,MAX_ALLOWED_MASS)[0])

def string_to_list(string):
    return_list = []
    for i in string:
        if i != " ":
            return_list.append(int(i))
    return return_list

def calculate_individual_fitness(individual_bin_configuration):
    calculated_mass = 0
    calculated_value = 0
    for i in range(0, NO_OF_QUBITS_INDIVIDUAL):
        if individual_bin_configuration[i]==1:
            calculated_mass += items[i].m
            calculated_value += items[i].v
    fitness_value = fitness_function(calculated_mass,calculated_value)
    if calculated_mass < MAX_ALLOWED_MASS:
        return True, fitness_value
    else:
        return False,fitness_value

def ufit():
    # define quantum registers
    ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL, "ind_qreg")
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS + 1, "fit_qreg")

    # create the U_fit subcircuit
    qc = QuantumCircuit(ind_qreg, fit_qreg, name="U$_fit$")

    controls = [ind_qreg[j] for j in range(NO_OF_QUBITS_INDIVIDUAL)]
    mcx = MCXGate(len(controls))  # reusable gate

    for i in range(POPULATION_SIZE):
        # prepare individual
        individual_binary = to_binary(i, NO_OF_QUBITS_INDIVIDUAL, True)
        for k in range(NO_OF_QUBITS_INDIVIDUAL):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])

        # compute fitness
        valid, fitness_value = calculate_individual_fitness(individual_binary)
        fitness_value_binary = to_binary(
            fitness_value, NO_OF_QUBITS_FITNESS, True
        )

        # write fitness value
        for k in range(NO_OF_QUBITS_FITNESS):
            if fitness_value_binary[k] == 1:
                qc.append(mcx, controls + [fit_qreg[k]])

        # mark validity bit
        if valid:
            qc.append(mcx, controls + [fit_qreg[NO_OF_QUBITS_FITNESS]])

        # uncompute individual
        for k in range(NO_OF_QUBITS_INDIVIDUAL):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])

        qc.barrier()

    return qc.to_instruction()


def oracle(max_value):
    # define and initialize fitness quantum register
    fit_reg = QuantumRegister(NO_OF_QUBITS_FITNESS + 1, "fqreg")
    # define and initialize max quantum register
    max_reg = QuantumRegister(NO_OF_QUBITS_FITNESS, "max")
    # define and initialize carry quantum register
    carry_in = QuantumRegister(1, "cin")
    carry_out = QuantumRegister(1, "cout")
    carry_help = QuantumRegister(1, "help")
    # define and initialize oracle workspace quantum register
    oracle = QuantumRegister(1, "oqreg")

    subtractor = CDKMRippleCarryAdder(NO_OF_QUBITS_FITNESS, kind='full')
    adder = CDKMRippleCarryAdder(NO_OF_QUBITS_FITNESS, kind='fixed')
    oracle_circ = QuantumCircuit(fit_reg, max_reg, carry_in, carry_out, carry_help, oracle, name="O")
    # Encode for subtraction

    oracle_circ.x(carry_in[0])
    oracle_circ.x(max_reg)

    oracle_circ.append(subtractor, [carry_in[0]] + max_reg[:] + fit_reg[:-1] + [carry_out[0]])

    oracle_circ.h(oracle[0])
    controls = [fit_reg[i] for i in range(NO_OF_QUBITS_FITNESS + 1)]
    mcx = MCXGate(len(controls))  # reusable gate
    oracle_circ.append(mcx, controls + [oracle[0]])
    oracle_circ.h(oracle[0])

    # oracle_circ.x(fit_reg[NO_OF_QUBITS_FITNESS-1])
    # oracle_circ.cx(fit_reg[NO_OF_QUBITS_FITNESS-1],oracle[0])
    # oracle_circ.x(fit_reg[NO_OF_QUBITS_FITNESS-1])

    oracle_circ.x(max_reg)
    oracle_circ.x(carry_in[0])
    oracle_circ.append(adder, max_reg[:] + fit_reg[:-1] + [carry_help[0]])
    # print(f"Oracle circuit:\n{oracle_circ.draw()}")
    return oracle_circ.to_instruction()


def grover():
    fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS + 1, "f")

    # create grover diffuser subcircuit
    grover_circ = QuantumCircuit(fit_qreg, name="U$_s$")

    grover_circ.h(fit_qreg)
    grover_circ.x(fit_qreg)

    grover_circ.h(fit_qreg[NO_OF_QUBITS_FITNESS])

    controls = [fit_qreg[i] for i in range(NO_OF_QUBITS_FITNESS)]
    mcx = MCXGate(len(controls))  # reusable gate
    grover_circ.append(mcx, controls + [fit_qreg[NO_OF_QUBITS_FITNESS]])

    grover_circ.h(fit_qreg[NO_OF_QUBITS_FITNESS])

    grover_circ.x(fit_qreg)
    grover_circ.h(fit_qreg)
    # print(f"Grover:\n{grover_circ.draw()}")

    return grover_circ.to_instruction()


def rqga():
    rqga_results = []
    max_value = 0
    new_fitness_value = get_random_value(0, 2 ** NO_OF_QUBITS_FITNESS)
    generation = 0


    while new_fitness_value != max_value and generation < NO_OF_GENERATIONS:
        ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL, "i")
        fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS + 1, "f")
        carry_in = QuantumRegister(1, "cin")
        carry_out = QuantumRegister(1, "cout")
        carry_help = QuantumRegister(1, "help")
        ws = QuantumRegister(1, "ws")
        max_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS, "max")
        creg = ClassicalRegister(NO_OF_QUBITS_FITNESS, "m")
        qc = QuantumCircuit(ind_qreg, fit_qreg, carry_in, carry_out, carry_help, max_qreg, ws, creg)

        qc.h(ind_qreg)
        qc.x(ws)
        qc.h(ws)

        ufit_gate = ufit()
        qc.append(ufit_gate, ind_qreg[:] + fit_qreg[:])
        max_value = new_fitness_value
        max_value_bin = to_binary(max_value, NO_OF_QUBITS_FITNESS, True)

        # Encode Max value into max_reg
        for i in range(0, NO_OF_QUBITS_FITNESS):
            if max_value_bin[i] == 1:
                qc.x(max_qreg[i])

        oracle_gate = oracle(max_value_bin)

        grover_gate = grover()
        for _ in range(0, NO_OF_MAX_GROVER_ITERATIONS):
            qc.append(oracle_gate, fit_qreg[:] + max_qreg[:] + carry_in[:] + carry_out[:] + carry_help[:] + ws[:])
            qc.append(grover_gate, fit_qreg[:])

        qc.barrier()
        for i in range(NO_OF_QUBITS_FITNESS):
            qc.measure(fit_qreg[i], creg[i])

        simulator = AerSimulator(method="statevector", device="GPU")

        qc = transpile(qc, simulator)
        results = simulator.run(qc).result()
        counts = results.get_counts()
        max_item = max(counts.items(), key=operator.itemgetter(1))
        new_fitness_value = int(max_item[0], 2)
        print(f"Found solution with fitness {new_fitness_value} in generation {generation} with {NO_OF_MAX_GROVER_ITERATIONS} grover iterations")
        rqga_results.append((generation, new_fitness_value))

        generation += 1
        # reset max value qreg
        for i in range(0, NO_OF_QUBITS_FITNESS):
            if max_value_bin[i] == 1:
                qc.x(max_qreg[i])
        # max_item = max(counts.items(), key = operator.itemgetter(1))
        # solution = string_to_list(max_item[0])
        # new_fitness_value = abs(calculate_individual_fitness(solution)[1])
        # print("Found solution {0} with fitness {1}...".format(max_item[0],new_fitness_value))
        # rqga_results.append((generation,new_fitness_value))
        # # get the max value
        # generation+=1
        # # reset max value qreg
        # for i in range(0, NO_OF_QUBITS_FITNESS):
        #     if max_value_bin[i] == 1:
        #         qc.x(max_qreg[i])
    return rqga_results

if __name__ == '__main__':
    final_results = rqga()
    print(f"Results: {final_results}")