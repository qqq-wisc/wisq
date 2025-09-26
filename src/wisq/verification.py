# File: optimization_compare.py
"""
Module for optimization metrics comparison on noiseless and noisy backends.
Usage:
    from optimization_compare import compare_optimization
    compare_optimization(bo_qasm_path, ao_qasm_path)
"""
from qiskit import qasm2, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import (
    FakeWashingtonV2, FakeFez, FakeMarrakesh, FakeTorino,
    FakeBrisbane, FakeCusco, FakeKawasaki, FakeKyiv,
    FakeOsaka, FakeQuebec
)
from qiskit.quantum_info.analysis import hellinger_fidelity
from typing import List, Tuple
from .utils import simulate_circuits, compute_errors, get_1q_2q_gate_counts, calculate_xeb

def _print_table(title: str, rows: List[Tuple[str, float]]):
    hdr = ("Metric", "Value")
    col1 = max(len(hdr[0]), *(len(label) for label, _ in rows))
    col2 = max(len(hdr[1]), *(len(str(val)) for _, val in rows))
    width = col1 + 3 + col2

    print(title.center(width, "-"))
    print(f"| {hdr[0]:<{col1}} | {hdr[1]:>{col2}} |")
    print(f"+-{'-'*col1}-+-{'-'*col2}-+")
    for label, val in rows:
        print(f"| {label:<{col1}} | {val:>{col2}} |")
    print(f"+-{'-'*col1}-+-{'-'*col2}-+")
    print()


def compare_optimization(bo_qasm_path: str,
                         ao_qasm_path: str,
                         fake_backends: List = None):
    """
    Load two QASM circuits, simulate, and print optimization metrics.
    """
    fake_backends = fake_backends or [
        FakeWashingtonV2(), FakeFez(), FakeMarrakesh(), FakeTorino(),
        FakeBrisbane(), FakeCusco(), FakeKawasaki(), FakeKyiv(),
        FakeOsaka(), FakeQuebec()
    ]
    aer_backend = AerSimulator()

    cir_BO = qasm2.load(bo_qasm_path)
    cir_AO = qasm2.load(ao_qasm_path)

    tbo, res_bo = simulate_circuits(cir_BO, backend=aer_backend)
    tao, res_ao = simulate_circuits(cir_AO, backend=aer_backend)

    fake = FakeWashingtonV2()
    fbo, fres_bo = simulate_circuits(cir_BO, backend=fake)
    fao, fres_ao = simulate_circuits(cir_AO, backend=fake)

    _, _, err_bo = compute_errors(fbo, backend=fake)
    _, _, err_ao = compute_errors(fao, backend=fake)

    bo_1q, bo_2q = get_1q_2q_gate_counts(tbo.count_ops())
    ao_1q, ao_2q = get_1q_2q_gate_counts(tao.count_ops())
    fq_bo_1, fq_bo_2 = get_1q_2q_gate_counts(fbo.count_ops())
    fq_ao_1, fq_ao_2 = get_1q_2q_gate_counts(fao.count_ops())

    xeb_bo = calculate_xeb(res_bo, cir_BO, fake_backends)
    xeb_ao = calculate_xeb(res_ao, cir_AO, fake_backends)

    opt_metrics = [
        ("Fidelity verification",hellinger_fidelity(res_bo, res_ao)),
        ("Error reduction on noisy backend",max(0,err_bo - err_ao)),
        ("Circuit depth reduction on noisless simulator",max(0,tbo.depth() - tao.depth()) ),
        ("Circuit depth reduction on noisy backend",max(0,fbo.depth() - fao.depth())),
        ("1-qubit gate reduction on noisless simulator",max(0,bo_1q - ao_1q)),
        ("2-qubit gate reduction on noisless simulator",max(0,bo_2q - ao_2q)),
        ("1-qubit gate reduction on noisy backend",max(0,fq_bo_1 - fq_ao_1)),
        ("2-qubit gate reduction on noisy backend",max(0,fq_bo_2 - fq_ao_2)),
    ]
    
    xeb_metrics = [
        ("XEB before optimization", xeb_bo),
        ("XEB after optimization", xeb_ao),
        ("XEB Improvement", max(0,xeb_ao - xeb_bo)),
    ]

    _print_table(" Optimization Verification & Comparison ", opt_metrics)
    _print_table(" Cross-Entropy Benchmarking ", xeb_metrics)
