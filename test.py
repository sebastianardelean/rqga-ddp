import numpy as np
import os
import math
import operator

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

def fitness_function(mass,value):
    return int(value - (TOTAL_VALUE+1)*divmod(mass,MAX_ALLOWED_MASS)[0])

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

def bin_to_dec(bit_str):
    bits = np.array(list(bit_str), dtype=int)  # convert to array
    decimal = np.dot(bits, 2 ** np.arange(bits.size))  # unsigned value
    decimal = decimal - (1 << bits.size) * (bits[-1])  # apply 2's complement if MSB=1
    return decimal

if __name__ == '__main__':
    for i in range(POPULATION_SIZE):
        individual_binary = to_binary(i, NO_OF_QUBITS_INDIVIDUAL, True)
        valid, fitness_value = calculate_individual_fitness(individual_binary)
        fitness_value_binary = to_binary(
            fitness_value, NO_OF_QUBITS_FITNESS, True
        )
        print(f"Individual: {i}->{individual_binary}, fitness {fitness_value}->{fitness_value_binary}->{bin_to_dec(''.join(str(b) for b in fitness_value_binary))} is {valid}")
