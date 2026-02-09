from itertools import permutations

def partitions_from_order(order):
    """
    Generate all possible contiguous partitions of an ordered list.
    Example: [1,2,3] →
      [[1],[2],[3]]
      [[1,2],[3]]
      [[1],[2,3]]
      [[1,2,3]]
    """
    n = len(order)
    for mask in range(1 << (n - 1)):
        part = []
        current = [order[0]]
        for i in range(n - 1):
            if mask & (1 << i):
                part.append(current)
                current = []
            current.append(order[i + 1])
        part.append(current)
        yield part


def sums_equal(partition, target_multiset):
    sums = sorted(sum(block) for block in partition)
    return sums == sorted(target_multiset)


def solve_ddp(A, B, C):
    solutions = []

    for perm in set(permutations(C)):
        for partA in partitions_from_order(perm):
            if not sums_equal(partA, A):
                continue

            for partB in partitions_from_order(perm):
                if not sums_equal(partB, B):
                    continue

                solutions.append({
                    "C_order": perm,
                    "A_partition": partA,
                    "B_partition": partB
                })

    return solutions


# ---- Run example ----

A = [6, 4]
B = [5, 5]
C = [1, 2, 3, 4]

solutions = solve_ddp(A, B, C)

for i, sol in enumerate(solutions, 1):
    print(f"\nSolution {i}")
    print("C order:", sol["C_order"])
    print("A partition:", sol["A_partition"], 
          "→", [sum(x) for x in sol["A_partition"]])
    print("B partition:", sol["B_partition"], 
          "→", [sum(x) for x in sol["B_partition"]])
