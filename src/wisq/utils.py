import os
from time import time_ns
import random
from typing import OrderedDict, List, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info.analysis import hellinger_fidelity
import numpy as np
from qiskit.providers import Backend
from qiskit.result import Counts


def create_scratch_dir(output_path: str) -> str:
    # Create temporary scratch directory for GUOQ
    timestamp = time_ns()
    uid = f"{timestamp}_{random.randint(0, 10000000)}"
    scratch_dir_name = f"wisq_tmp_{uid}"
    scratch_dir_path = os.path.join(os.path.dirname(output_path), scratch_dir_name)
    os.mkdir(scratch_dir_path)
    return (scratch_dir_path, uid)

def calculate_xeb(ideal_counts: OrderedDict, circ: QuantumCircuit, bknds: List) -> float:
    xeb_scores = []
    # Iterate through each fake backend
    for bknd in bknds:
        _, noisy_counts = simulate_circuits(circ,backend=bknd,shots=1024)
        xeb_scores.append(hellinger_fidelity(ideal_counts, noisy_counts))

    # Return the average XEB fidelity
    return float(np.mean(xeb_scores))

def get_1q_2q_gate_counts(ops: OrderedDict) -> Tuple[float,float]:
    one_q_gate_count = 0
    two_q_gate_count = 0
    for key,value in ops.items():
        if key not in ['cx','cz','cy','measure','barrier','unitary']:
            one_q_gate_count += value
        elif key in ['cx','cz','cy']:
            two_q_gate_count += value
        else:
            continue
    return one_q_gate_count,two_q_gate_count

def compute_errors(circuit: QuantumCircuit, backend: Backend) -> Tuple[float, float, float]:
    """
    Compute the total gate error and total readout error for a transpiled circuit on a given backend.

    Parameters:
        circuit (QuantumCircuit): The transpiled quantum circuit.
        backend (Backend): The backend to extract error information from.

    Returns:
        Tuple[float, float, float]: A tuple containing total_gate_error, total_readout_error, and their sum.
    """
    backend_props = backend.properties()
    total_gate_error = 0.0
    readout_qubits = set()

    for instruction in circuit.data:
        op = instruction.operation
        qubits = instruction.qubits

        # Extract qubit indices using find_bit
        try:
            qubit_indices = [circuit.find_bit(q).index for q in qubits]
        except AttributeError:
            # For Qiskit versions where find_bit is not available
            qubit_indices = [circuit.qubits.index(q) for q in qubits]

        readout_qubits.update(qubit_indices)

        # Retrieve gate error from backend properties
        try:
            gate_error = backend_props.gate_error(op.name, qubit_indices)
            total_gate_error += gate_error
        except Exception:
            # Handle cases where gate error information is unavailable
            pass

    total_readout_error = 0.0
    for qubit in readout_qubits:
        try:
            readout_error = backend_props.readout_error(qubit)
            total_readout_error += readout_error
        except Exception:
            # Handle cases where readout error information is unavailable
            pass

    return (total_gate_error + total_readout_error), total_readout_error, total_gate_error

def simulate_circuits(
    circuit: QuantumCircuit,
    backend: Backend,
    shots: int = 1024,
) -> Tuple[QuantumCircuit, Counts]:
    """
    Simulate a quantum circuit on a specified backend with optional noise model.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to simulate.
        backend (Backend): The backend to simulate the circuit on.
        shots (int): Number of shots for the simulation.

    Returns:
        Tuple[QuantumCircuit, Counts]: A tuple containing the transpiled circuit and the result counts.
    """
    # Ensure the circuit has measurements
    circuit = circuit.copy()
    circuit.remove_final_measurements()
    circuit.measure_all()
    transpiled_circuit = transpile(circuit, backend=backend,optimization_level=3)
    job = backend.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return transpiled_circuit, counts