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




# PROBLEM SETUP (Section II A)
# A, B and C are multisets of fragment lengths in increasing order:
# A = {a_1, ..., a_m}, B = {b_1, ..., b_n}, C = {c_1, ..., c_k}
# with sum(A) = sum(B) = sum(C).
# The mapping sequences mu and nu are permutations of the indices
# [1..m] and [1..n]; the goal is to find mu and nu such that the
# reconstructed multiset C'_{mu,nu} equals C. With partial cleavage
# error, the objective is min sum |C'_i - c_i|.

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
NO_OF_QUBITS_FITNESS = 10
# The paper defines f(mu,nu) = 1 / (1 + sum|C'_i - c_i|), with optimum 1.
# The RQGA encodes fitness on NO_OF_QUBITS_FITNESS qubits, so scale the
# paper's fitness onto the largest representable integer.
MAX_FITNESS = 2 ** NO_OF_QUBITS_FITNESS - 1

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


def cleavage_error(mu, nu) -> int:
    # mapping sequences mu, nu reorder the fragments of A and B
    A_perm = [A[i] for i in mu]
    B_perm = [B[i] for i in nu]
    C_mod = list(TARGET)

    k = len(C_mod)
    expected_k = SIZE_A + SIZE_B - 1

    # --- REPAIR SEQUENCE LOGIC (Section II B) ---
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

    # union of the accumulative summations: both end at sum(A) = sum(B),
    # so exactly one copy of the shared final total is kept
    merged = sorted(A_cum[:-1] + B_cum)

    # step difference of the union yields C_{mu,nu}; sorted in
    # increasing order it is C'_{mu,nu}
    steps = step_difference(merged)

    C_prime = sorted(steps)

    error = sum(abs(s - t) for s, t in zip(C_prime, sorted(C_mod)))
    return error


def fitness_function(solution) -> (bool,int):
    # chromosome encoding: [ u | ν ]
    valid,mu,nu = decode_solution(solution)
    if valid == False:
        return False, 0

    error = cleavage_error(mu, nu)
    # integer encoding of the paper's fitness 1 / (1 + error)
    return True, MAX_FITNESS // (1 + error)


def optimization_fitness_function(solution)->(bool,int):
    # chromosome encoding: [ u | ν ]
    valid, mu, nu = decode_solution(solution)
    if valid == False:
        return False, 0

    return True, cleavage_error(mu, nu)

if __name__ == '__main__':
    # Run the RQGA to reduce the search space

    final_results = rqga(NO_OF_QUBITS_INDIVIDUAL,NO_OF_QUBITS_FITNESS,NO_OF_GENERATIONS,fitness_function,1)
    print(f"Results: {final_results}")
