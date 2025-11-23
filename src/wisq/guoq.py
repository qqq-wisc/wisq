import subprocess
import multiprocessing
import os
import shutil
import glob
import requests
import random
import sys
import platform
from time import time_ns
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary as sel
from qiskit import qasm2
from .utils import create_scratch_dir
from .resynth import start_server
from .qualtran_rotation_synthesis import QualtranRS

GUOQ_JAR = os.path.join(
    os.path.dirname(__file__), "lib", "GUOQ-1.0-jar-with-dependencies.jar"
)
RULES_DIR = os.path.join(os.path.dirname(__file__), "lib", "rules")


CLIFFORDT = "CLIFFORDT"
FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE = "FT"
GATE_SETS = {
    "NAM": ["rz", "h", "x", "cx"],
    CLIFFORDT: ["t", "tdg", "s", "sdg", "h", "x", "cx"],
    "IBMO": ["u1", "u2", "u3", "cx"],
    "IBMN": ["rz", "sx", "x", "cx"],
    "ION": ["rx", "ry", "rz", "rxx"],
}

ERROR_BUDGET = 2


def start_resynth_server(bqskit=False, verbose=False, path_to_synthetiq=None):
    p = multiprocessing.Process(
        target=start_server, args=(bqskit, True, verbose, path_to_synthetiq)
    )
    p.start()
    return p


def is_server_ready():
    try:
        response = requests.get("http://localhost:8080")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def print_help():
    command = f"java -ea -cp {GUOQ_JAR} qoptimizer.Optimizer -h"
    command_list = command.split(" ")
    proc = subprocess.Popen(
        command_list,
    )
    proc.wait()


def write_args_file(args, args_file, circuit_file):
    with open(args_file, "w") as f:
        for k, v in args.items():
            f.write(f"{k}\n")
            if v is not None:
                f.write(f"{v}\n")
        f.write(f"{circuit_file}\n")


def transpile_if_needed(
    input_path, target_gateset, scratch_dir, approximation_epsilon=0
):
    circuit = QuantumCircuit.from_qasm_file(input_path)
    approximation = 0

    # Check if need to transpile
    gates = set(circuit.count_ops().keys())
    need_to_transpile = False
    for gate in gates:
        if gate not in GATE_SETS[target_gateset]:
            need_to_transpile = True

    if not need_to_transpile:
        return (approximation, input_path)

    transpiled = None
    if target_gateset == CLIFFORDT:
        pm = PassManager(
            [BasisTranslator(equivalence_library=sel, target_basis=GATE_SETS["NAM"])]
        )
        nam_circuit = pm.run(circuit)
        num_rz = nam_circuit.count_ops().get("rz", 0)
        print(f"Decomposing to Clifford + T using Qualtran rotation synthesis. Estimated time: {10*num_rz} seconds.")
        approximation_per_angle = approximation_epsilon / (num_rz * ERROR_BUDGET)
        approximation = approximation_epsilon / ERROR_BUDGET

        pm = PassManager([QualtranRS(approximation_per_angle)])

        transpiled = pm.run(nam_circuit)
    else:
        pm = PassManager(
            [
                BasisTranslator(
                    equivalence_library=sel, target_basis=GATE_SETS[target_gateset]
                )
            ]
        )
        transpiled = pm.run(circuit)

    output_path = os.path.join(
        scratch_dir, f"transpiled_{time_ns()}_" + os.path.basename(input_path)
    )
    qasm2.dump(transpiled, output_path)
    return (approximation, output_path)


def run_guoq(
    input_path,
    output_path,
    target_gateset,
    optimization_objective,
    timeout=3600,
    approximation_epsilon=0,
    args=None,
    verbose=False,
    path_to_synthetiq=None,
):
    # Create temporary scratch directory for GUOQ
    scratch_dir_path, uid = create_scratch_dir(output_path)

    try:
        (approximation, transpiled_path) = transpile_if_needed(
            input_path, target_gateset, scratch_dir_path, approximation_epsilon
        )
        approximation_epsilon = approximation_epsilon - approximation

        if timeout == 0:
            shutil.move(transpiled_path, output_path)
            return

        # Write GUOQ args to file
        args_file_path = os.path.join(scratch_dir_path, f"args_{uid}.txt")
        input_args = args
        args = {}
        args["--rules-dir"] = RULES_DIR
        args["-g"] = target_gateset
        args["-opt"] = optimization_objective
        if approximation_epsilon == 0:
            args["-resynth"] = "NONE"
        else:
            args["-eps"] = approximation_epsilon
        if verbose:
            args["--verbosity"] = 2
        if input_args is not None:
            args.update(input_args)
        args["-out"] = scratch_dir_path
        args["-job"] = uid
        write_args_file(args, args_file_path, transpiled_path)

        # Start resynthesis server if needed
        resynth_proc = None
        if args.get("-resynth", None) != "NONE":
            if optimization_objective in ["FT", "T"] and path_to_synthetiq is None:
                system = platform.system().lower()
                processor = platform.processor().lower()
                if system == "linux" and processor in ["x86_64"]:
                    path_to_synthetiq = f"./bin/main_linux_{processor}"
                elif system == "darwin" and processor in ["arm", "i386"]:
                    path_to_synthetiq = f"./bin/main_mac_{processor}"
                else:
                    print(
                        "Unsupported platform for pre-compiled Synthetiq. Please follow the instructions here to compile Synthetiq for your platform: https://github.com/eth-sri/synthetiq/tree/bbe3c1299a97295f5af38eec647f6bbe9fdd9234. Then try again using the `--abs_path_to_synthetiq/-apts` option to pass in the absolute path to the Synthetiq `bin/main` binary."
                    )
                    sys.exit(1)
            resynth_proc = start_resynth_server(
                bqskit="BQSKIT" in args.values()
                or optimization_objective in ["TWO_Q", "FIDELITY"],
                verbose=verbose,
                path_to_synthetiq=path_to_synthetiq,
            )
            # Wait for server to spin up
            while not is_server_ready():
                continue

        # Invoke GUOQ
        command = f"java -ea -cp {GUOQ_JAR} qoptimizer.Optimizer @{args_file_path}"
        command_list = command.split(" ")
        proc = subprocess.Popen(
            command_list,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()

        # Kill resynthesis server
        if resynth_proc is not None:
            resynth_proc.terminate()
            resynth_proc.join()
    finally:
        for source_file in glob.glob(
            os.path.join(scratch_dir_path, f"latest*{uid}*qasm")
        ):
            shutil.move(source_file, output_path)
        # Clean up scratch directory
        if os.path.exists(scratch_dir_path):
            shutil.rmtree(scratch_dir_path)
