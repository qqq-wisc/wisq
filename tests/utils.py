import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
import os
import time
from qiskit.circuit.random import random_circuit
from qiskit.qasm2 import dump as dump_qasm_2
from qiskit import transpile
from pathlib import Path

def assert_no_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        helper: checks for measurements in quantum circuit

        args:
            circuit: QuantumCircuit
                the input quantum circuit
        returns:
            QuantumCircuit: the same circuit as passed in
        throws:
            ValueError: if the input circuit contains measurements
        """
        for instruction in circuit.data:
            if instruction.operation.name == "measure":
                raise ValueError("circuit contains measurements")
        return circuit


def is_circuit_equiv(qasm1: QuantumCircuit, qasm2: QuantumCircuit):
    """
    compare two quantum circuits to check if they are equal to each other,
    up to some global phase

    this piggybacks on the qiskit Operation's equiv(), which finds the entire unitary
    of the circuit. thus, it's really slow and memory-intensive, as the matrices
    will be 2^n by 2^n, but it is exact

    try to use with <= 12 qubits. anything over that might throw due to memory
    issues

    args:
        qasm1, qasm2: QuantumCircuit
            the input quantum circuits to check
    returns:
        bool: True if equal, False ow
    throws:
        ValueError: if either of the input circuits contain measurements
    """
    assert_no_measurements(qasm1)
    assert_no_measurements(qasm2)

    if qasm1.num_qubits != qasm2.num_qubits:
        return False

    op1 = Operator(qasm1)
    op2 = Operator(qasm2)
    return op1.equiv(op2)


def is_circuit_equiv_random_sv(qasm1: QuantumCircuit, qasm2: QuantumCircuit, tol: float = 1e-8, trials: int = 5) -> bool:
    """
    compare two quantum circuits to check if they are equal to each other,
    up to some global phase

    this is done by basically fuzzing both circuits, and checking that the
    resulting states are within a certain tolerance of each other
    (up to a global phase)

    this scales a lot better then the official implementation, but it is probabilistic,
    albeit barely. the odds of a random space lining up with an eigenvector
    get exponentially smaller the more qubits we have and is basically 0

    this still scales really badly, because there will be 2^n different states
    but it still scales much better than finding the unitary of the entire circuit
    (which is what qiskit's does)

    still, try to only use with <= 28 qubits. anything over that might throw
    due to memory issues

    args:
        qasm1, qasm2: QuantumCircuit
            the input quantum circuits to check
        tol: float
            tolerance for equality
        trials: int
            the number of test circuits to pump through
    returns:
        bool: True if equal within tolerance, False ow
    throws:
        ValueError: if either of the input circuits contain measurements
    """
    assert_no_measurements(qasm1)
    assert_no_measurements(qasm2)

    if qasm1.num_qubits != qasm2.num_qubits:
        return False
    n = qasm1.num_qubits
    for _ in range(trials):
        psi = np.random.randn(2**n) + 1j * np.random.randn(2**n)
        psi /= np.linalg.norm(psi)
        sv = Statevector(psi)
        out1 = sv.evolve(qasm1)
        out2 = sv.evolve(qasm2)
        # check for global phase
        inner = np.vdot(out1.data, out2.data)
        if abs(abs(inner) - 1.0) > tol:
            return False
    
    return True


def build_random_qasm(
        num_qubits: int,
        depth: int,
        basis_gates: list[str] | None = None,
        dir_name: str = "randomly-generated-circuits",
        file_name: str | None = None
    ) -> None:
    """
    builds a random qasm file with the given params.
    basically a wrapper for qiskit.circuit.random's random_circuit,
    except it also saves the respective .qasm file
    
    args:
        num_qubits: int
            the number of qubits
        depth: int
            the depth of the generated circuit
        basis_gates: list[str] | None = None
            the list of gates to be included in the circuit
            if left as None, the gates will be selected from
            qiskit.circuit.library.standard_gates
        dir_name: str = "randomly-generated-circuits"
            the name of the directory to store the randomly generated circuits
        file_name: str | None
            the name to give the generated file. if left blank,
            a generated name with the num_qubits, depth, and time of
            creation will be used
    output:
        None
            writes file to folder randomly-generated-circuits
    """
    
    circ = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
    )
    circ = transpile(circ, basis_gates=basis_gates, optimization_level=0)
    os.makedirs(dir_name, exist_ok=True)
    if file_name == None:
        file_name = f"random_{num_qubits}q_{depth}d-{int(time.time())}"
    output_path = Path(dir_name) / f"{file_name}.qasm"
    dump_qasm_2(circ, output_path)
