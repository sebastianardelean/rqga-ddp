import pygad
import numpy as np

A = sorted([6, 4])
B = sorted([5, 5])
C = [1, 2, 3, 4]

n = len(C)

# ---- Helpers ----

def apply_cuts(order, cuts):
    blocks = []
    current = [order[0]]
    for i, cut in enumerate(cuts):
        if cut == 1:
            blocks.append(current)
            current = []
        current.append(order[i + 1])
    blocks.append(current)
    return [sum(block) for block in blocks]


def fitness_func(ga, solution, solution_idx):
    perm = solution[:n].astype(int)
    cutA = solution[n:n + n - 1].astype(int)
    cutB = solution[n + n - 1:].astype(int)

    # invalid permutation → kill it
    if len(set(perm)) != n:
        return -1e6

    order = [C[i] for i in perm]

    sumsA = sorted(apply_cuts(order, cutA))
    sumsB = sorted(apply_cuts(order, cutB))

    # penalties
    penalty = 0
    penalty += sum(abs(a - b) for a, b in zip(
        sumsA + [0] * (len(A) - len(sumsA)), A))
    penalty += sum(abs(a - b) for a, b in zip(
        sumsB + [0] * (len(B) - len(sumsB)), B))

    # exact match bonus
    bonus = 0
    if sumsA == A:
        bonus += 1000
    if sumsB == B:
        bonus += 1000

    return bonus - penalty


# ---- Gene space ----

gene_space = (
    [list(range(n))] * n +        # permutation genes
    [[0, 1]] * (n - 1) +           # A cuts
    [[0, 1]] * (n - 1)             # B cuts
)

# ---- GA setup ----

ga = pygad.GA(
    num_generations=300,
    num_parents_mating=10,
    sol_per_pop=30,
    num_genes=len(gene_space),
    gene_space=gene_space,
    fitness_func=fitness_func,
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10,
    keep_parents=5
)

# ---- Run GA ----

ga.run()

solution, fitness, _ = ga.best_solution()

# ---- Decode solution ----

perm = solution[:n].astype(int)
cutA = solution[n:n + n - 1].astype(int)
cutB = solution[n + n - 1:].astype(int)

order = [C[i] for i in perm]

print("Best fitness:", fitness)
print("C order:", order)
print("A fragments:", sorted(apply_cuts(order, cutA)))
print("B fragments:", sorted(apply_cuts(order, cutB)))
