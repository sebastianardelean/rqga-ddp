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
from fnmatch import fnmatch


def select_simulator() -> AerSimulator:
    # Prefer the tensor-network method (GPU when available). Some qiskit-aer
    # builds are compiled without tensor_network support, so probe it first
    # and fall back to statevector instead of failing at runtime.
    method = "tensor_network"
    try:
        available_devices = AerSimulator(method=method).available_devices()
    except Exception:
        method = "statevector"
        print("tensor_network method unavailable, falling back to statevector")
        available_devices = AerSimulator(method=method).available_devices()
    if "GPU" not in available_devices:
        print(f"GPU device unavailable ({available_devices}), falling back to CPU")
    return AerSimulator(
        method=method, device="GPU" if "GPU" in available_devices else "CPU"
    )


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
    # The fitness value is encoded on no_of_qubits_fitness + 1 qubits
    # (two's complement, sign bit included) plus one validity qubit.
    ind_qreg = QuantumRegister(no_of_qubits_individual, "ind_qreg")
    fit_qreg = QuantumRegister(no_of_qubits_fitness + 2, "fit_qreg")

    # create the U_fit subcircuit
    qc = QuantumCircuit(ind_qreg, fit_qreg, name="U_fit")

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
            fitness_value, no_of_qubits_fitness + 1, True
        )

        # write fitness value (sign bit included)
        for k in range(no_of_qubits_fitness + 1):
            if fitness_value_binary[k] == 1:
                qc.append(mcx, controls + [fit_qreg[k]])

        # mark validity bit
        if valid:
            qc.append(mcx, controls + [fit_qreg[no_of_qubits_fitness + 1]])
        # uncompute individual
        for k in range(no_of_qubits_individual):
            if individual_binary[k] == 0:
                qc.x(ind_qreg[k])

        qc.barrier()

    return qc.to_instruction()


def oracle(max_value, no_of_qubits_fitness:int, no_of_qubits_individual:int) -> Instruction:
    # Fitness register: value field on no_of_qubits_fitness + 1 qubits
    # (f_M ... f_0, two's complement) followed by the validity qubit.
    value_bits = no_of_qubits_fitness + 1
    fit_reg = QuantumRegister(value_bits + 1, "fqreg")
    max_reg = QuantumRegister(value_bits, "max")
    carry_in = QuantumRegister(1, "cin")
    carry_out = QuantumRegister(1, "cout")
    sign_copy = QuantumRegister(1, "signcopy")
    oracle = QuantumRegister(1, "oqreg")

    subtractor = CDKMRippleCarryAdder(value_bits, kind='full')
    oracle_circ = QuantumCircuit(
        fit_reg, max_reg, carry_in, carry_out, sign_copy, oracle, name="O"
    )

    value = fit_reg[:value_bits]
    sign = fit_reg[value_bits - 1]
    ports = [carry_in[0]] + max_reg[:] + value + [carry_out[0]]

    # Remember whether the stored fitness is a non-negative value (sign 0).
    oracle_circ.cx(sign, sign_copy[0])

    # value := value + ~max + 0 = value - max - 1 (mod 2**(M+1)).
    # For a non-negative stored value the sign bit of the result is 0 exactly
    # when value > max.
    oracle_circ.x(max_reg)
    oracle_circ.append(subtractor, ports)

    # Phase-flip marked states (the oracle workspace qubit is in |->).
    oracle_circ.x(sign)
    oracle_circ.x(sign_copy[0])
    oracle_circ.append(
        MCXGate(2), [sign, sign_copy[0], oracle[0]]
    )
    oracle_circ.x(sign_copy[0])
    oracle_circ.x(sign)

    # Uncompute the subtraction, restoring value, max, carry_in and carry_out.
    oracle_circ.append(subtractor.inverse(), ports)
    oracle_circ.x(max_reg)
    oracle_circ.cx(sign, sign_copy[0])

    return oracle_circ.to_instruction()


def diffuser(no_of_qubits:int) -> Instruction:
    # Grover diffuser (inversion about the uniform superposition) on the
    # searched qubits. The fitness register stays entangled with the
    # individual register, so the amplification has to act on the individual
    # (respectively free) qubits; diffusing the fitness register alone cannot
    # change its measurement statistics.
    qreg = QuantumRegister(no_of_qubits, "d")

    diffuser_circ = QuantumCircuit(qreg, name="Diff")
    if no_of_qubits == 0:
        return diffuser_circ.to_instruction()

    diffuser_circ.h(qreg)
    diffuser_circ.x(qreg)

    if no_of_qubits == 1:
        diffuser_circ.z(qreg[0])
    else:
        diffuser_circ.h(qreg[no_of_qubits - 1])
        controls = [qreg[i] for i in range(no_of_qubits - 1)]
        diffuser_circ.append(
            MCXGate(len(controls)), controls + [qreg[no_of_qubits - 1]]
        )
        diffuser_circ.h(qreg[no_of_qubits - 1])

    diffuser_circ.x(qreg)
    diffuser_circ.h(qreg)

    return diffuser_circ.to_instruction()


def grover(no_of_qubits_fitness:int) -> Instruction:
    fit_qreg = QuantumRegister(no_of_qubits_fitness + 1, "f")

    # create grover diffuser subcircuit
    grover_circ = QuantumCircuit(fit_qreg, name="Diff")

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
        no_of_fixed_genes: int = 0,
        no_of_grover_iterations: int = None,
) -> list[(int, int)]:
    rqga_results = []
    max_value = initial_max_value
    generation = 0

    population_size = 2**no_of_qubits_individual

    simulator = select_simulator()

    while generation < no_of_generations:
        # while generation < 1:
        print("Creating the circuit!")
        ind_qreg = QuantumRegister(no_of_qubits_individual, "i")
        fit_qreg = QuantumRegister(no_of_qubits_fitness + 2, "f")
        carry_in = QuantumRegister(1, "cin")
        carry_out = QuantumRegister(1, "cout")
        sign_copy = QuantumRegister(1, "signcopy")
        ws = QuantumRegister(1, "ws")
        max_qreg = QuantumRegister(no_of_qubits_fitness + 1, "max")
        creg = ClassicalRegister(no_of_qubits_individual, "m")
        qc = QuantumCircuit(ind_qreg, fit_qreg, carry_in, carry_out, sign_copy, max_qreg, ws, creg)

        marked = 0
        for i in range(population_size):
            individual = to_binary(i, no_of_qubits_individual, True)
            valid, value = fitness_function(individual)
            if valid and value > max_value:
                marked += 1
        if marked == 0:
            break
        if no_of_grover_iterations is None:
            iterations = max(
                0,
                int(round(math.pi / 4 * math.sqrt(population_size / marked) - 0.5)),
            )
        else:
            iterations = no_of_grover_iterations
        print(f"Marked individuals: {marked}/{population_size}, using {iterations} grover iteration(s)")

        print("Creating the superposition")
        qc.h(ind_qreg)
        qc.x(ws)
        qc.h(ws)

        if iterations > 0:
            print("Applying ufit")
            ufit_gate = ufit(population_size,no_of_qubits_individual, no_of_qubits_fitness,fitness_function)
            max_value_bin = to_binary(max_value, no_of_qubits_fitness + 1, True)

            print("Encoding max value")
            for i in range(0, no_of_qubits_fitness + 1):
                if max_value_bin[i] == 1:
                    qc.x(max_qreg[i])

            print("Creating the oracle gate")
            oracle_gate = oracle(max_value_bin,no_of_qubits_fitness, no_of_qubits_individual)
            print("Creating grover gate")
            diffuser_gate = diffuser(no_of_qubits_individual)

            for i in range(iterations):
                print(f"Starting iteration {i}")
                print("\tApply ufit")
                qc.append(ufit_gate, ind_qreg[:] + fit_qreg[:])
                print("\tApply Grover's oracle")
                qc.append(oracle_gate, fit_qreg[:] + max_qreg[:] + carry_in[:] + carry_out[:] + sign_copy[:] + ws[:])
                print("\tUncompute ufit")
                qc.append(ufit_gate.inverse(), ind_qreg[:] + fit_qreg[:])
                print("\tApply Grover's diffuser")
                qc.append(diffuser_gate, ind_qreg[:])

        qc.barrier()
        print("Measure the circuit")
        qc.measure(ind_qreg, creg)

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
            individual = to_binary(int(max_item[0], 2), no_of_qubits_individual, True)
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
