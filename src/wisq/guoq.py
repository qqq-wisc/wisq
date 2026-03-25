import subprocess
import multiprocessing
import os
import shutil
import glob
import requests
import signal
import sys
import platform
import time
from time import time_ns
from copy import copy
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn

_console = Console()
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary as sel
from qiskit import qasm2
from .utils import create_scratch_dir, split_circuit

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
    from .resynth import start_server

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
    

def get_resynth_proc(args, optimization_objective, verbose, path_to_synthetiq):
    resynth_proc = None
    if args.get("-resynth", None) != "NONE":
        if optimization_objective in ["FT", "T"] and path_to_synthetiq is None:
            system = platform.system().lower()
            processor = platform.processor().lower() or platform.machine().lower()
            if system == "linux" and processor in ["x86_64"]:
                path_to_synthetiq = f"./bin/main_linux_{processor}"
            elif system == "darwin" and processor in ["arm", "i386"]:
                path_to_synthetiq = f"./bin/main_mac_{processor}"
            else:
                _console.print(
                    "[bold red]Error:[/bold red] Unsupported platform for pre-compiled Synthetiq. "
                    "Please compile Synthetiq for your platform according to "
                    "https://github.com/eth-sri/synthetiq/tree/bbe3c1299a97295f5af38eec647f6bbe9fdd9234 "
                    "and pass the [bold]bin/main[/bold] binary absolute path via "
                    "[bold]--abs_path_to_synthetiq[/bold] or [bold]-apts[/bold]."
                )
                sys.exit(1)
        resynth_proc = start_resynth_server(
            bqskit="BQSKIT" in args.values()
            or optimization_objective in ["TWO_Q", "FIDELITY"],
            verbose=verbose,
            path_to_synthetiq=path_to_synthetiq,
        )
        # Wait for server to spin up
        with _console.status("    [dim]Starting resynthesis server...[/dim]"):
            while not is_server_ready():
                time.sleep(0.1)
    
    return resynth_proc


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
    if target_gateset == CLIFFORDT: # TODO: update when https://github.com/qqq-wisc/wisq/pull/34 merged
        if approximation_epsilon == 0:
            _console.print(
                            "[bold red]Error:[/bold red] Decomposing to Clifford + T requires non-zero approximation epsilon. "
                            "Please pass a value strictly between 0 and 1 to "
                            "[bold]--approx_epsilon[/bold] or [bold]-ap[/bold]."
                        )
            sys.exit(1)

        pm = PassManager(
            [BasisTranslator(equivalence_library=sel, target_basis=GATE_SETS["NAM"])]
        )
        nam_circuit = pm.run(circuit)
        num_rz = nam_circuit.count_ops().get("rz", 0)
        _console.print(f"    [dim]Decomposing to Clifford + T via Qualtran rotation synthesis  [bold]~{10*num_rz}s[/bold][/dim]")
        approximation_per_angle = approximation_epsilon / (num_rz * ERROR_BUDGET)
        approximation = approximation_epsilon / ERROR_BUDGET

        from .qualtran_rotation_synthesis import QualtranRS

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


def run_guoq_single(
    input_path,
    output_path,
    target_gateset,
    timeout,
    approximation_epsilon,
    args,
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
        extended_args = copy(args)
        extended_args["-out"] = scratch_dir_path
        extended_args["-job"] = uid
        write_args_file(extended_args, args_file_path, transpiled_path)

        # Invoke GUOQ
        command = f"java -ea -cp {GUOQ_JAR} qoptimizer.Optimizer @{args_file_path}"
        command_list = command.split(" ")
        proc = subprocess.Popen(
            command_list,
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("    [dim #7C3AED]{task.description}[/dim #7C3AED]"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            console=_console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Optimizing  (timeout: {timeout}s)", total=timeout)
            start = time.time()
            while proc.poll() is None:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    proc.terminate()
                    break
                progress.update(task, completed=int(elapsed))
                time.sleep(0.5)
            progress.update(task, completed=timeout)
    finally:
        for source_file in glob.glob(
            os.path.join(scratch_dir_path, f"latest*{uid}*qasm")
        ):
            shutil.move(source_file, output_path)
        # Clean up scratch directory
        if os.path.exists(scratch_dir_path):
            shutil.rmtree(scratch_dir_path)


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
    threads=1,
): 
    assert threads >= 1, "must split into >= 1 chunks to optimize"

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

    # Start resynthesis server if needed
    resynth_proc = get_resynth_proc(args, optimization_objective, verbose, path_to_synthetiq)
    scratch_dir_path = None

    try:
        if threads == 1:
            run_guoq_single(
                input_path,
                output_path,
                target_gateset,
                timeout,
                approximation_epsilon,
                args,
            )
        else:
            scratch_dir_path, _ = create_scratch_dir(output_path)
            chunk_paths = split_circuit(input_path, threads, scratch_dir_path)

            # Build optimized output paths: "optimized_" prepended to chunk filename
            optimized_paths = [
                os.path.join(scratch_dir_path, f"optimized_{os.path.basename(p)}")
                for p in chunk_paths
            ]

            # Optimize each chunk in parallel
            processes = []
            for chunk_path, opt_path in zip(chunk_paths, optimized_paths):
                p = multiprocessing.Process(
                    target=run_guoq_single,
                    args=(
                        chunk_path,
                        opt_path,
                        target_gateset,
                        timeout,
                        approximation_epsilon,
                        args,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # Stitch optimized chunks back together in order
            combined = QuantumCircuit.from_qasm_file(optimized_paths[0])
            for opt_path in optimized_paths[1:]:
                chunk = QuantumCircuit.from_qasm_file(opt_path)
                combined.compose(chunk, inplace=True)

            qasm2.dump(combined, output_path)
                
    finally:
        # Prevent a second KeyboardInterrupt from aborting cleanup
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Terminate any still-running child processes
        if threads > 1:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            for p in processes:
                p.join(timeout=5)

        # Clean up scratch directory
        if scratch_dir_path is not None and os.path.exists(scratch_dir_path):
            shutil.rmtree(scratch_dir_path, ignore_errors=True) # handles subdirectories

        # Kill resynthesis server
        if resynth_proc is not None:
            resynth_proc.terminate()
            resynth_proc.join()

        signal.signal(signal.SIGINT, old_handler)
    