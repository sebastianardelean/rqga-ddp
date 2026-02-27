# DDP Problem

import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import os
import math

from rqga import rqga

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
from qiskit.circuit.library import FullAdderGate


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


MAX_ERROR = 627
# ENCODING

BITS_M = math.ceil(math.log2(SIZE_A))
BITS_N = math.ceil(math.log2(SIZE_B))
NO_OF_QUBITS_INDIVIDUAL = NO_BITS_RANK_A + NO_BITS_RANK_B
NO_OF_QUBITS_FITNESS = 10

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
    return True,  max(0, MAX_ERROR - error)

def string_to_list(string):
    return_list = []
    for i in string:
        if i != " ":
            return_list.append(int(i))
    return return_list

def optimization_fitness_function(solution)->(bool,int):
    # chromosome encoding: [ u | ν ]
    valid, mu, nu = decode_solution(solution)
    if valid == False:
        return False, 0

    # Check if valid permutations (exact set match)
    if set(mu) != set(range(SIZE_A)) or set(nu) != set(range(SIZE_B)):
        return False, 0  # invalid permutation

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
    return True, error

if __name__ == '__main__':
    # Run the RQGA to reduce the search space

    gene_individual = 10
    gene_fitness = 10
    no_of_generations = 10




    final_results = rqga(NO_OF_QUBITS_INDIVIDUAL,NO_OF_QUBITS_FITNESS,NO_OF_GENERATIONS,fitness_function,0)
    print(f"Results: {final_results}")
