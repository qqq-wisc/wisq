from bqskit import compile, Circuit, MachineModel
from bqskit.ir.gates import (
    CXGate,
    RZGate,
    HGate,
    XGate,
    RXGate,
    RYGate,
    RXXGate,
    U1Gate,
    U2Gate,
    U3Gate,
    SXGate,
)
from bqskit.compiler import Compiler
from bqskit.ext import bqskit_to_qiskit
from bqskit.ext import qiskit_to_bqskit
from qiskit import QuantumCircuit
from qiskit import qasm2
import argparse
import urllib.parse
import socketserver
from http.server import BaseHTTPRequestHandler
import warnings
import time
import subprocess
import os
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary as sel
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
)
import shutil
import numpy as np
from qiskit.quantum_info import Operator
import json
import platform
from functools import partial
import sys

LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")

# begin code from https://github.com/eth-sri/synthetiq/blob/main/notebooks/post_processing/analyzer.py
NON_STANDARD_GATES = {
    "scz": (
        "crz(pi)",
        Operator(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1j]])),
    ),
    "U": (
        "crx(pi)",
        Operator(
            np.array(
                [
                    [
                        -0.35355339 + 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 - 0.35355339j,
                    ],
                    [
                        0.35355339 - 0.35355339j,
                        0.35355339 + 0.35355339j,
                        -0.35355339 - 0.35355339j,
                        0.35355339 - 0.35355339j,
                    ],
                    [
                        0.35355339 - 0.35355339j,
                        -0.35355339 - 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 - 0.35355339j,
                    ],
                    [
                        0.35355339 - 0.35355339j,
                        0.35355339 + 0.35355339j,
                        0.35355339 + 0.35355339j,
                        -0.35355339 + 0.35355339j,
                    ],
                ]
            )
        ),
    ),
}


class Circuit:
    def __init__(self, filename) -> None:
        with open(filename, "r") as file:
            qasm_str = file.read()
        for replace_gate in NON_STANDARD_GATES:
            qasm_str = qasm_str.replace(
                replace_gate, NON_STANDARD_GATES[replace_gate][0]
            )
        self.circuit = QuantumCircuit.from_qasm_str(qasm_str)
        for index, gate in enumerate(self.circuit.data):
            for replace_gate in NON_STANDARD_GATES:
                if (
                    gate[0].name == NON_STANDARD_GATES[replace_gate][0].split("(")[0]
                    and gate[0].params[0] == np.pi
                ):
                    # set gate to the correct gate
                    self.circuit.data[index] = (
                        NON_STANDARD_GATES[replace_gate][1],
                        gate[1],
                        gate[2],
                    )

        self.filename = filename
        self.score = float(os.path.basename(filename).split("-")[0])
        self.t_depth = self.circuit.depth(lambda gate: gate[0].name in ["t", "tdg"])
        self.cx_depth = self.circuit.depth(lambda gate: gate[0].name == "cx")
        self.cx_count = np.count_nonzero(
            np.array([el[0].name for el in self.circuit.data]) == "cx"
        )
        gates_names = np.array([el[0].name for el in self.circuit.data])
        self.t_count = np.count_nonzero(gates_names == "t") + np.count_nonzero(
            gates_names == "tdg"
        )
        self.count = float(os.path.basename(filename).split("-")[1])
        self.gates = len(gates_names)


def main_analysis(circuit_folder):
    t_depth = []
    t_count = []
    gates = []
    best_t_depth_circ = None
    best_t_count_circ = None
    best_cx_depth_circ = None
    for file in os.listdir(circuit_folder):
        circuit = Circuit(os.path.join(circuit_folder, file))
        t_depth.append(circuit.t_depth)
        gates.append(circuit.gates)
        t_count.append(circuit.t_count)

        # if best_t_depth_circ is None:
        #     best_t_depth_circ = circuit
        # condition2 = circuit.t_depth < best_t_depth_circ.t_depth
        # condition3 = (
        #     circuit.t_depth == best_t_depth_circ.t_depth
        #     and circuit.t_count < best_t_depth_circ.t_count
        # )
        # condition4 = (
        #     circuit.t_depth == best_t_depth_circ.t_depth
        #     and circuit.t_count == best_t_depth_circ.t_count
        #     and circuit.score < best_t_depth_circ.score
        # )
        # if condition2 or condition3 or condition4:
        #     best_t_depth_circ = circuit

        if best_t_count_circ is None:
            best_t_count_circ = circuit
        condition2 = circuit.t_count < best_t_count_circ.t_count
        condition3 = (
            circuit.t_count == best_t_count_circ.t_count
            and circuit.t_depth < best_t_count_circ.t_depth
        )
        condition4 = (
            circuit.t_count == best_t_count_circ.t_count
            and circuit.t_depth == best_t_count_circ.t_depth
            and circuit.score < best_t_count_circ.score
        )
        if condition2 or condition3 or condition4:
            best_t_count_circ = circuit

        # do the same for cx depth
        # if best_cx_depth_circ is None:
        #     best_cx_depth_circ = circuit
        # condition2 = circuit.cx_depth < best_cx_depth_circ.cx_depth
        # condition3 = (
        #     circuit.cx_depth == best_cx_depth_circ.cx_depth
        #     and circuit.cx_count < best_cx_depth_circ.cx_count
        # )
        # condition4 = (
        #     circuit.cx_depth == best_cx_depth_circ.cx_depth
        #     and circuit.cx_count == best_cx_depth_circ.cx_count
        #     and circuit.score < best_cx_depth_circ.score
        # )
        # if condition2 or condition3 or condition4:
        #     best_cx_depth_circ = circuit

    t_depth = np.array(t_depth)
    t_count = np.array(t_count)
    gates = np.array(gates)
    return (
        t_depth,
        t_count,
        gates,
        best_t_count_circ,
        best_t_depth_circ,
        best_cx_depth_circ,
    )


# end code from https://github.com/eth-sri/synthetiq/blob/main/notebooks/post_processing/analyzer.py

warnings.filterwarnings("ignore")

GATE_SET_DICT = {
    "ibm_new": ["cx", "rz", "sx", "x"],
    "nam": ["cx", "rz", "h", "x"],
    "ion": {RXXGate(), RZGate(), RXGate(), RYGate()},
}


def bqskit_io(compiler, data, circuit_str, opt_level, epsilon, target_gateset):
    qc = QuantumCircuit.from_qasm_str(circuit_str)
    data["circuit"] = circuit_str
    data["original_size"] = qc.size()
    data["original_2q_size"] = qc.num_nonlocal_gates()
    model = (
        MachineModel(qc.num_qubits, gate_set=GATE_SET_DICT[target_gateset])
        if target_gateset == "ion"
        else None
    )
    circuit = compile(
        qiskit_to_bqskit(qc).get_unitary(),
        optimization_level=opt_level,
        synthesis_epsilon=epsilon,
        compiler=compiler,
        model=model,
    )
    data["bqskit_params"] = {"opt_level": opt_level, "epsilon": epsilon}
    data["resynth_size"] = circuit.num_operations
    data["resynth_2q_size"] = (
        circuit.gate_counts[CXGate()] if CXGate() in circuit.gate_counts else 0
    ) + (circuit.gate_counts[RXXGate()] if RXXGate() in circuit.gate_counts else 0)

    if target_gateset != "none" and target_gateset != "ion":
        circuit = bqskit_to_qiskit(circuit)
        pm = PassManager(
            [
                BasisTranslator(sel, GATE_SET_DICT[target_gateset]),
            ]
        )
        circuit = pm.run(circuit)
        return qasm2.dumps(circuit)

    return circuit.to("qasm")


def get_t_count(circuit):
    count_ops = circuit.count_ops()
    return count_ops.get("t", 0) + count_ops.get("tdg", 0)


def synthetiq_disk(
    data,
    circuit_str,
    num_circuits,
    epsilon,
    threads,
    target_gateset,
    verbose,
    path_to_synthetiq,
):
    qc = QuantumCircuit.from_qasm_str(circuit_str)
    matrix = Operator(qc).data
    data["circuit"] = circuit_str
    data["original_size"] = qc.size()
    data["original_t_size"] = get_t_count(qc)
    data["original_2q_size"] = qc.num_nonlocal_gates()

    temp_circ = f"circ_{np.random.randint(0, 1000000)}"

    with open(f"{LIB_DIR}/synthetiq/data/input/{temp_circ}.txt", "w") as f:
        f.write(f"{temp_circ}\n")
        f.write(f"{qc.num_qubits}\n")
        for row in matrix:
            for val in row:
                f.write(f"({val.real},{val.imag}) ")
            f.write("\n")
        for row in matrix:
            for val in row:
                f.write(f"1 ")
            f.write("\n")

    directory = f"{LIB_DIR}/synthetiq/data/output/{temp_circ}"
    temp_circ_path = f"{LIB_DIR}/synthetiq/data/input/{temp_circ}.txt"

    command = f"{path_to_synthetiq} {temp_circ}.txt -c {num_circuits} -eps {epsilon} -h {threads}"
    data["synthetiq_command"] = command
    command_list = command.split(" ")
    proc = subprocess.Popen(
        command_list,
        cwd=os.path.join(LIB_DIR, "synthetiq"),
        stdout=subprocess.DEVNULL if not verbose else None,
        stderr=subprocess.DEVNULL if not verbose else None,
    )
    proc.wait()

    (
        t_depth,
        t_count,
        gates,
        best_t_count_circ,
        best_t_depth_circ,
        best_cx_depth_circ,
    ) = main_analysis(directory)

    if os.path.exists(directory):
        shutil.rmtree(directory)
    if os.path.exists(temp_circ_path):
        os.remove(temp_circ_path)

    new_circ = best_t_count_circ.circuit
    data["resynth_size"] = new_circ.size()
    data["resynth_t_size"] = get_t_count(new_circ)
    data["resynth_2q_size"] = new_circ.num_nonlocal_gates()

    if target_gateset != "none":
        pm = PassManager(
            [
                BasisTranslator(sel, GATE_SET_DICT[target_gateset]),
            ]
        )
        new_circ = pm.run(new_circ)

    # rename qreg because synthetiq uses "qubits" by default
    return qasm2.dumps(new_circ).replace("qubits[", qc.qregs[0].name + "[")


class MyHandler(BaseHTTPRequestHandler):
    def __init__(
        self, bqskit, bqskit_auto_workers, verbose, path_to_synthetiq, *args, **kwargs
    ):
        self.compiler = None
        if bqskit:
            if bqskit_auto_workers:
                self.compiler = Compiler()
            else:
                self.compiler = Compiler(num_workers=64)
        self.verbose = verbose
        self.path_to_synthetiq = path_to_synthetiq
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/bqskit":
            parsed_body = json.loads(body)
            time1 = time.time()
            data = {}
            output = bqskit_io(
                self.compiler,
                data,
                parsed_body["circuit"],
                int(parsed_body["opt_level"]),
                float(parsed_body["epsilon"]),
                parsed_body["target_gateset"],
            )
            data["resynthesized_circuit"] = output
            time2 = time.time()
            data["time"] = time2 - time1
            print(data)
        if parsed_path.path == "/synthetiq":
            parsed_body = json.loads(body)
            time1 = time.time()
            data = {}
            output = synthetiq_disk(
                data,
                parsed_body["circuit"],
                int(parsed_body["num_circuits"]),
                float(parsed_body["epsilon"]),
                int(parsed_body["threads"]),
                parsed_body["target_gateset"],
                self.verbose,
                self.path_to_synthetiq,
            )
            data["resynthesized_circuit"] = output
            time2 = time.time()
            data["time"] = time2 - time1
            print(data)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(output.encode("utf-8"))

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()


def start_server(bqskit, bqskit_auto_workers, verbose=False, path_to_synthetiq=None):
    if not verbose:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    partial_handler = partial(
        MyHandler, bqskit, bqskit_auto_workers, verbose, path_to_synthetiq
    )
    try:
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", 8080), partial_handler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "--bqskit",
        action=argparse.BooleanOptionalAction,
        help="Use BQSKit (initialize BQSKit compiler instance). Not necessary if using some other resynthesis algorithm (e.g. Synthetiq).",
    )
    parser.add_argument(
        "--bqskit_auto_workers",
        action=argparse.BooleanOptionalAction,
        help="[Recommended] Use BQSKit default mechanism for determining how many workers to spin up",
    )
    parser.add_argument(
        "--path_to_synthetiq",
        type=str,
        help="Absolute path to Synthetiq `main` binary",
        default=os.path.join(LIB_DIR, "synthetiq", "bin", "main"),
    )

    args = parser.parse_args()

    start_server(
        args.bqskit,
        args.bqskit_auto_workers,
        verbose=True,
        path_to_synthetiq=args.path_to_synthetiq,
    )
