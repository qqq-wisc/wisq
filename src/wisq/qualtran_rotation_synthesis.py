from qiskit.transpiler import PassManager, TranspilerError, TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit import qasm2
from qiskit import QuantumCircuit
import mpmath
from qualtran.rotation_synthesis import math_config as mc
from qualtran.rotation_synthesis.protocols import clifford_t_synthesis as cts
import sys

def sequence_to_circ(sequence : str) -> QuantumCircuit:
    circ = QuantumCircuit(1)
    for gate in sequence:
        if gate == "S":
            circ.s(0)
        elif gate == "H":
            circ.h(0)
        elif gate == "Tx":
            circ.h(0)
            circ.t(0)
            circ.h(0)
        elif gate == "Ty":
            circ.sdg(0)
            circ.h(0)
            circ.t(0)
            circ.h(0)
            circ.s(0)
        elif gate == "Tz":
            circ.t(0)
        elif gate == "X":
            circ.x(0)
        elif gate == "Y":
            circ.sdg(0)
            circ.x(0)
            circ.s(0)
        elif gate == "Z":
            circ.h(0)
            circ.x(0)
            circ.h(0)
    return circ

class QualtranRS(TransformationPass):

    def __init__(self, epsilon=1e-10) -> None:
        """
        Approximately decompose 1q gates to a discrete basis using Qualtran's implementation of [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064).
        Args:
        epsilon : the permitted error of approximation
        """
        super().__init__()
        self.approx_exp = mpmath.mpf(epsilon)
        self.qualtran_rs_config = mc.with_dps(200) # good for up to 10-20? increasing makes it slower
        self.max_t = 400

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ``QualtranRS`` pass on `dag`.

        Args:
            dag: The input dag.

        Returns:
            Output dag with 1q gates synthesized in the discrete target basis.
        """
        for node in dag.op_nodes():
            if not node.name == "rz":
                continue  # ignore all non-rz qubit gates

            angle = node.op.params[0]

            diagonal = cts.diagonal_unitary_approx(theta=angle, eps=self.approx_exp, max_n=self.max_t, config=self.qualtran_rs_config)

            if diagonal is None:
                raise TranspilerError(f"Could not decompose rotation by angle {angle} within approximation epsilon {self.approx_exp} and max T-count {self.max_t}.")

            sequence = diagonal.to_matrix().to_sequence()

            decomposed = sequence_to_circ(sequence)

            approx_dag = circuit_to_dag(decomposed)

            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, approx_dag)

        return dag
