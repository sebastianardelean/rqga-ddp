# DDP Problem

import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import os
import math
import operator
# import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools


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



# PROBLEM SETUP

A = [7, 8, 18]
B = [1,2,2,4,5,5,5,9]
C = [1,2,2,2,2,3,4,5,5,7]

SIZE_A = len(A)
SIZE_B = len(B)
TARGET = sorted(C)

NO_RANKS_A = math.factorial(SIZE_A)
NO_RANKS_B = math.factorial(SIZE_B)

NO_BITS_RANK_A = math.ceil(math.log2(NO_RANKS_A))
NO_BITS_RANK_B = math.ceil(math.log2(NO_RANKS_B))

# ENCODING

BITS_M = math.ceil(math.log2(SIZE_A))
BITS_N = math.ceil(math.log2(SIZE_B))
NO_OF_QUBITS_INDIVIDUAL = NO_BITS_RANK_A + NO_BITS_RANK_B
NO_OF_QUBITS_FITNESS = 8

POPULATION_SIZE = 2**NO_OF_QUBITS_INDIVIDUAL
NO_OF_MAX_GROVER_ITERATIONS = round(math.pi/4 * math.sqrt(2**NO_OF_QUBITS_FITNESS) - 0.5)
NO_OF_GENERATIONS = 10






# Helper functions

def rank_perm(p):
    n = len(p)
    rank = 0
    for i in range(n):
        smaller = sum(p[j] < p[i] for j in range(i+1, n))
        rank += smaller * math.factorial(n - i - 1)
    return rank

def unrank_perm(n, rank):
    import math

    elements = list(range(n))
    perm = []

    for i in range(n-1, -1, -1):
        fact = math.factorial(i)
        idx = rank // fact
        rank %= fact
        perm.append(elements.pop(idx))

    return perm

# ----------------------------
# Decode binary chromosome
# ----------------------------
def decode_solution(solution):
    rank_a_bits = solution[:NO_BITS_RANK_A]
    rank_b_bits = solution[NO_BITS_RANK_A:]
    rank_a = int(''.join(map(str, rank_a_bits)), 2)
    rank_b = int(''.join(map(str, rank_b_bits)), 2)
    if rank_a >= NO_RANKS_A or rank_a < 0:
        return False,[],[]
    if rank_b >= NO_RANKS_B or rank_b < 0:
        return False,[],[]
    #print(f"rank_a: {rank_a}, rank_b: {rank_b}")
    perm_a = unrank_perm(SIZE_A, rank_a)
    perm_b = unrank_perm(SIZE_B, rank_b)
    return True, perm_a,perm_b

def cumulative_sum(arr):
    return np.cumsum(arr).tolist()


def step_difference(arr):
    return [arr[0]] + [arr[i] - arr[i-1] for i in range(1, len(arr))]





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



def fitness_function(solution) -> (bool,int):
    # chromosome encoding: [ u | ν ]
    valid,mu,nu = decode_solution(solution)
    if valid == False:
        return False, 0


    # Check if valid permutations (exact set match)
    if set(mu) != set(range(SIZE_A)) or set(nu) != set(range(SIZE_B)):
        return False,0  # invalid permutation

    A_perm = [A[i] for i in mu]
    B_perm = [B[i] for i in nu]
    C_mod = list(TARGET)

    k = len(C_mod)
    expected_k = SIZE_A + SIZE_B - 1

    # --- REPAIR SEQUENCE LOGIC ---
    if k < expected_k:
        # Rule 1: Add 0-length fragments to target C
        delta = expected_k - k
        C_mod.extend([0] * delta)

    elif k > expected_k:
        # Rule 2: Add 0-length fragments to the shorter sequence among A and B
        delta = k - expected_k
        if SIZE_A <= SIZE_B:
            A_perm.extend([0] * delta)
        else:
            B_perm.extend([0] * delta)

    A_cum = cumulative_sum(A_perm)
    B_cum = cumulative_sum(B_perm)

    merged = sorted(A_cum + B_cum)

    steps = step_difference(merged)

    steps_sorted = sorted(set(steps))



    error = sum(abs(s - t) for s, t in zip(steps_sorted, TARGET))
#    fitness = 1.0 / (1.0 + error)
#    print(f"mu: {mu}; nu:{nu} -> fitness: {fitness}")
    return True, error

def string_to_list(string):
    return_list = []
    for i in string:
        if i != " ":
            return_list.append(int(i))
    return return_list



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
        valid, fitness_value = fitness_function(individual_binary)
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

    subtractor = CDKMRippleCarryAdder(NO_OF_QUBITS_FITNESS , kind='full')
    adder = CDKMRippleCarryAdder(NO_OF_QUBITS_FITNESS , kind='fixed')
    oracle_circ = QuantumCircuit(fit_reg, max_reg, carry_in, carry_out, carry_help, oracle, name="O")
    # Encode for subtraction

    oracle_circ.x(carry_in[0])
    oracle_circ.x(max_reg)

    oracle_circ.append(subtractor, [carry_in[0]] + max_reg[:] + fit_reg[:-1] + [carry_out[0]])


    # flip ancilla if fitreg = 0. After subtraction, if max is the actual fitness value, then the result in fitreg = 0 -> solution
    # keep in mind that the valid is 1 if the individual is valid.
    # Therefore, if fitness is the target, all the bits should be 0, except valid. 
    oracle_circ.x(fit_reg)
    oracle_circ.x(fit_reg[NO_OF_QUBITS_FITNESS])
    controls = [fit_reg[i] for i in range(NO_OF_QUBITS_FITNESS + 1)]
    mcx = MCXGate(len(controls))  # reusable gate
    oracle_circ.append(mcx, controls + [oracle[0]])
    oracle_circ.x(fit_reg[NO_OF_QUBITS_FITNESS]) 
    oracle_circ.x(fit_reg)

    #oracle_circ.h(oracle[0])
    #controls = [fit_reg[i] for i in range(NO_OF_QUBITS_FITNESS + 1)]
    #mcx = MCXGate(len(controls))  # reusable gate
    #oracle_circ.append(mcx, controls + [oracle[0]])
    #oracle_circ.h(oracle[0])
    
    #oracle_circ.x(fit_reg[NO_OF_QUBITS_FITNESS-1])
    #oracle_circ.cz(fit_reg[NO_OF_QUBITS_FITNESS-1], oracle[0])
    #oracle_circ.x(fit_reg[NO_OF_QUBITS_FITNESS-1])



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

def bin_to_dec(bit_str):
    bits = np.array(list(bit_str), dtype=int)  # convert to array
    decimal = np.dot(bits, 2 ** np.arange(bits.size))  # unsigned value
    decimal = decimal - (1 << bits.size) * (bits[-1])  # apply 2's complement if MSB=1
    return int(decimal)

def bin_to_dec_msb(bit_str):
    bits = np.array(list(bit_str), dtype=int)           # covert to array
    decimal = np.dot(bits[::-1], 2 ** np.arange(bits.size))  # treat LSB at index 0
    decimal = decimal - (1 << bits.size) * bits[0]     # subtract 2^n if MSB=1
    return int(decimal)


def rqga():
    rqga_results = []
    max_value = -1
    new_fitness_value = 0
    generation = 0


    while new_fitness_value != max_value and generation < NO_OF_GENERATIONS:
    #while generation < 1:
        print("Creating the circuit!")
        ind_qreg = QuantumRegister(NO_OF_QUBITS_INDIVIDUAL, "i")
        fit_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS + 1, "f")
        carry_in = QuantumRegister(1, "cin")
        carry_out = QuantumRegister(1, "cout")
        carry_help = QuantumRegister(1, "help")
        ws = QuantumRegister(1, "ws")
        max_qreg = QuantumRegister(NO_OF_QUBITS_FITNESS , "max")
        creg = ClassicalRegister(NO_OF_QUBITS_FITNESS, "m")
        qc = QuantumCircuit(ind_qreg, fit_qreg, carry_in, carry_out, carry_help, max_qreg, ws, creg)

        print("Creating the superposition")
        qc.h(ind_qreg)
        qc.x(ws)
        qc.h(ws)

        print("Applying ufit")
        ufit_gate = ufit()
        qc.append(ufit_gate, ind_qreg[:] + fit_qreg[:])
        max_value = new_fitness_value
        max_value_bin = to_binary(max_value, NO_OF_QUBITS_FITNESS , True)

        print("Encoding max value")
        # Encode Max value into max_reg
        for i in range(0, NO_OF_QUBITS_FITNESS):
            if max_value_bin[i] == 1:
                qc.x(max_qreg[i])

        print("Creating the oracle gate")
        oracle_gate = oracle(max_value_bin)
        print("Creating grover gate")
        grover_gate = grover()
        #for _ in range(0, NO_OF_MAX_GROVER_ITERATIONS):
        
        for _ in range(0,1):
            print("Apply grover")
            qc.append(oracle_gate, fit_qreg[:] + max_qreg[:] + carry_in[:] + carry_out[:] + carry_help[:] + ws[:])
            qc.append(grover_gate, fit_qreg[:])

        qc.barrier()
        for i in range(NO_OF_QUBITS_FITNESS):
            qc.measure(fit_qreg[i], creg[i])
        #qc.measure(fit_qreg,creg)
        simulator = AerSimulator(method="statevector", device="GPU")

        #qc = transpile(qc, simulator)
        results = simulator.run(qc).result()
        counts = results.get_counts()
        max_item = max(counts.items(), key=operator.itemgetter(1))
        print(f"Counts are: {counts} and max is:{max_item}")
        new_fitness_value = bin_to_dec_msb(max_item[0])
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
