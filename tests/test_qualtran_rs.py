from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.quantum_info import diamond_norm, Choi, SuperOp
from wisq.qualtran_rotation_synthesis import QualtranRS
import mpmath
from qualtran.rotation_synthesis import math_config as mc
from qualtran.rotation_synthesis.channels import UnitaryChannel


def test_qualtran_rs():
    epsilon = 1e-10
    theta = 0.3

    circuit = QuantumCircuit(1)
    circuit.rz(theta, 0)
    pm = PassManager([QualtranRS(1e-10)])
    transpiled = pm.run(circuit)

    assert transpiled.count_ops().keys() <= {"t", "tdg", "s", "sdg", "h", "x", "cx"}
    sequence = []
    for gate in transpiled.data:
        if gate.operation.name == "t":
            sequence.append("Tz")
        elif gate.operation.name == "s":
            sequence.append("S")
        elif gate.operation.name == "sdg":
            sequence.append("S")
            sequence.append("S")
            sequence.append("S")
        elif gate.operation.name == "h":
            sequence.append("H")
        elif gate.operation.name == "x":
            sequence.append("X")
    
    assert UnitaryChannel.from_sequence(sequence).diamond_norm_distance_to_rz(theta, mc.with_dps(200)) <= mpmath.mpf(epsilon)