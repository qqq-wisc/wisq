from qiskit.transpiler import PassManager, TranspilerError, TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit import qasm2
from qiskit import QuantumCircuit
import mpmath
from qualtran.rotation_synthesis import math_config as mc
from qualtran.rotation_synthesis.protocols import clifford_t_synthesis as cts
from qualtran.rotation_synthesis.matrix import clifford_t_repr as ctr
from concurrent.futures import ProcessPoolExecutor
import sys


def _decompose_rz_worker(args: tuple) -> QuantumCircuit:
    """Worker for parallel Rz decomposition. Must be module-level for pickling."""
    angle, eps_float, max_n, dps = args
    config = mc.with_dps(dps)
    approx_exp = mpmath.mpf(eps_float)
    diagonal = cts.diagonal_unitary_approx(
        theta=angle, eps=approx_exp, max_n=max_n, config=config
    )
    if diagonal is None:
        raise ValueError(
            f"Could not decompose rotation by angle {angle} within "
            f"approximation epsilon {eps_float} and max T-count {max_n}."
        )
    sequence = ctr.to_sequence(diagonal.to_matrix())
    return sequence_to_circ(sequence)


def sequence_to_circ(sequence: str) -> QuantumCircuit:
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
        rz_nodes = [
            (node, node.op.params[0])
            for node in dag.op_nodes()
            if node.name == "rz"
        ]
        if not rz_nodes:
            return dag

        nodes, angles = zip(*rz_nodes)
        # Use float(approx_exp) so args are picklable for ProcessPoolExecutor
        eps_float = float(self.approx_exp)
        dps = 200  # must match self.qualtran_rs_config
        args_list = [(a, eps_float, self.max_t, dps) for a in angles]

        try:
            with ProcessPoolExecutor() as executor:
                circuits = list(executor.map(_decompose_rz_worker, args_list))
        except ValueError as e:
            raise TranspilerError(str(e)) from e

        for node, circ in zip(nodes, circuits):
            approx_dag = circuit_to_dag(circ)
            dag.substitute_node_with_dag(node, approx_dag)

        return dag
