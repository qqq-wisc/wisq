# File: resource_estimation.py
"""
Module for Q# resource estimation comparison.
Usage:
    from resource_estimation import compare_resources
    compare_resources(params_path, bo_qasm_path, ao_qasm_path)
"""
from qiskit import qasm2
from qsharp.interop.qiskit import estimate
import json


def _print_resource_table(title: str, rows):
    hdr = ("Metric", "Before optimization", "After optimization")
    col_widths = [
        max(len(hdr[i]), *(len(str(r[i])) for r in rows))
        for i in range(3)
    ]
    width = sum(col_widths) + 6

    print(title.center(width, "-"))
    print(f"| {hdr[0]:<{col_widths[0]}} | {hdr[1]:>{col_widths[1]}} | {hdr[2]:>{col_widths[2]}} |")
    print(f"+-{'-'*col_widths[0]}-+-{'-'*col_widths[1]}-+-{'-'*col_widths[2]}-+")
    for metric, bo, ao in rows:
        print(f"| {metric:<{col_widths[0]}} | {bo:>{col_widths[1]}} | {ao:>{col_widths[2]}} |")
    print(f"+-{'-'*col_widths[0]}-+-{'-'*col_widths[1]}-+-{'-'*col_widths[2]}-+")
    print()


def compare_resources(params_path: str, bo_qasm_path: str, ao_qasm_path: str):
    """
    Load two QASM circuits, run Q# resource estimation, and print comparison.
    """
    with open(params_path) as f:
        params = json.load(f)

    cir_BO = qasm2.load(bo_qasm_path)
    cir_AO = qasm2.load(ao_qasm_path)

    BO_res = estimate(cir_BO, params)
    AO_res = estimate(cir_AO, params)

    rows = [
        ("Physical Qubits", BO_res['physicalCountsFormatted']['physicalQubits'],
                             AO_res['physicalCountsFormatted']['physicalQubits']),
        ("Runtime",          BO_res['physicalCountsFormatted']['runtime'],
                             AO_res['physicalCountsFormatted']['runtime']),
        ("Clifford error rate", BO_res['physicalCounts']['breakdown']['cliffordErrorRate'],
                                 AO_res['physicalCounts']['breakdown']['cliffordErrorRate']),
        ("Logical Depth",    BO_res['physicalCountsFormatted']['logicalDepth'],
                             AO_res['physicalCountsFormatted']['logicalDepth']),
        ("T states",         BO_res['physicalCountsFormatted']['numTstates'],
                             AO_res['physicalCountsFormatted']['numTstates']),
    ]

    _print_resource_table(" Resource Estimates ", rows)
