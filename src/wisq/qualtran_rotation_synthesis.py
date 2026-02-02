from qiskit.transpiler import PassManager, TranspilerError, TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit
import mpmath
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler.passes import BasisTranslator
from qualtran.rotation_synthesis import math_config as mc
from qualtran.rotation_synthesis.protocols import clifford_t_synthesis as cts
from qualtran.rotation_synthesis.matrix import clifford_t_repr as ctr

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
        self._pi_over_two = mpmath.pi / 2
        self.clifford_pm = PassManager(
            BasisTranslator(
                SessionEquivalenceLibrary,
                ["id", "x", "y", "z", "h", "s", "sdg"],
            )
        )

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

            # Use mpmath for consistent high-precision comparison when detecting multiples of π/2.
            angle = mpmath.mpf(node.op.params[0])
            remainder = mpmath.fmod(angle, self._pi_over_two)
            is_clifford = mpmath.almosteq(remainder, 0, abs_eps=1e-12) or mpmath.almosteq(
                mpmath.fabs(remainder) - self._pi_over_two, 0, abs_eps=1e-12
            )
            if is_clifford:
                # Rz(k*π/2) for integer k: build Clifford circuit directly (BasisTranslator
                # does not have Rz in its equivalence library for these angles).
                equiv = mpmath.fmod(angle, 2 * mpmath.pi)
                k = int(mpmath.nint(equiv / self._pi_over_two)) % 4
                base = QuantumCircuit(1)
                if k == 1:
                    base.s(0)
                elif k == 2:
                    base.z(0)
                elif k == 3:
                    base.sdg(0)
                # k == 0: identity, no gates
                dag.substitute_node_with_dag(node, circuit_to_dag(base))
                continue

            diagonal = cts.diagonal_unitary_approx(
                theta=angle,
                eps=self.approx_exp,
                max_n=self.max_t,
                config=self.qualtran_rs_config,
            )

            if diagonal is None:
                raise TranspilerError(f"Could not decompose rotation by angle {angle} within approximation epsilon {self.approx_exp} and max T-count {self.max_t}.")

            sequence = ctr.to_sequence(diagonal.to_matrix())

            decomposed = sequence_to_circ(sequence)

            approx_dag = circuit_to_dag(decomposed)

            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, approx_dag)

        return dag
