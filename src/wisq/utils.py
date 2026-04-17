import os
from time import time_ns
import random


def create_scratch_dir(output_path: str) -> str:
    # Create temporary scratch directory for GUOQ
    timestamp = time_ns()
    uid = f"{timestamp}_{random.randint(0, 10000000)}"
    scratch_dir_name = f"wisq_tmp_{uid}"
    scratch_dir_path = os.path.join(os.path.dirname(output_path), scratch_dir_name)
    os.mkdir(scratch_dir_path)
    return (scratch_dir_path, uid)


def split_circuit(circuit_path, num_chunks, out_directory):
    from qiskit import QuantumCircuit, qasm2
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.dagcircuit import DAGCircuit

    circuit = QuantumCircuit.from_qasm_file(circuit_path)
    dag = circuit_to_dag(circuit)

    ops = list(dag.topological_op_nodes())
    chunk_size = len(ops) // num_chunks
    remainder = len(ops) % num_chunks

    basename = os.path.splitext(os.path.basename(circuit_path))[0]
    ext = os.path.splitext(circuit_path)[1]

    paths = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_ops = ops[start:end]
        start = end

        chunk_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            chunk_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            chunk_dag.add_creg(creg)
        for node in chunk_ops:
            chunk_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        chunk_circuit = dag_to_circuit(chunk_dag)
        out_path = os.path.join(out_directory, f"{basename}_chunk{i}{ext}")
        qasm2.dump(chunk_circuit, out_path)
        paths.append(out_path)

    return paths
