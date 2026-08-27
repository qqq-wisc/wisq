import subprocess
import multiprocessing
import os
import shutil
import glob
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

LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")
RULES_DIR = os.path.join(LIB_DIR, "rules")
SYNTHETIQ_BIN_DIR = os.path.join(LIB_DIR, "synthetiq", "bin")


CLIFFORDT = "CLIFFORDT"
FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE = "FT"
GATE_SETS = {
    "NAM": ["rz", "h", "x", "cx"],
    CLIFFORDT: ["t", "tdg", "s", "sdg", "h", "x", "cx"],
    "IBMO": ["u1", "u2", "u3", "cx"],
    "IBMN": ["rz", "sx", "x", "cx"],
    "ION": ["rx", "ry", "rz", "rxx"],
}

# GUOQ's built-in defaults for symbolic rules point at a `generated/` corpus that is
# synthesized by `queso` and not vendored here; pass the reference corpus we do ship
# explicitly instead.
SYMB_RULES = {
    "NAM": "rules_q3_s3_nam_symb.txt",
    CLIFFORDT: "rules_q3_s3_cliffordt_symb.txt",
    "IBMO": "rules_q3_s3_ibm_symb.txt",
    "IBMN": "rules_q3_s3_ibmnew_symb.txt",
    "ION": "rules_q3_s3_ion_symb.txt",
}

ERROR_BUDGET = 2


def _platform_binary(prefix, bin_dir):
    """The bundled binary for this platform, or None if there is none."""
    system = platform.system().lower()
    processor = platform.processor().lower() or platform.machine().lower()
    if system == "linux" and processor in ["x86_64"]:
        name = f"{prefix}_linux_{processor}"
    elif system == "darwin" and processor in ["arm", "i386"]:
        name = f"{prefix}_mac_{processor}"
    else:
        return None
    path = os.path.join(bin_dir, name)
    return path if os.path.exists(path) else None


def find_guoq_binary():
    """Locate the GUOQ optimizer binary.

    Resolution order: the WISQ_GUOQ_BIN environment variable, then the `guoq` pip
    package (a wisq dependency), then a `guoq` on PATH.
    """
    env_path = os.environ.get("WISQ_GUOQ_BIN")
    if env_path:
        if os.path.exists(env_path):
            return env_path
        _console.print(
            f"[bold red]Error:[/bold red] WISQ_GUOQ_BIN points to [bold]{env_path}[/bold], "
            "which does not exist."
        )
        sys.exit(1)
    try:
        from guoq import find_guoq_bin

        return find_guoq_bin()
    except (ImportError, FileNotFoundError):
        pass
    on_path = shutil.which("guoq")
    if on_path is not None:
        return on_path
    _console.print(
        "[bold red]Error:[/bold red] No GUOQ binary found. Install it with "
        "[bold]pip install guoq[/bold], or build the [bold]rust-port[/bold] branch of "
        "https://github.com/qqq-wisc/guoq with [bold]cargo build --release[/bold] and put "
        "[bold]target/release/guoq[/bold] on your PATH or point "
        "[bold]WISQ_GUOQ_BIN[/bold] at it."
    )
    sys.exit(1)


def backend_args(args, optimization_objective, path_to_synthetiq):
    """Paths GUOQ needs to drive its resynthesis backends.

    GUOQ owns its backends' lifecycles — it spawns Synthetiq or the BQSKit worker
    itself — so wisq only has to say where they are.
    """
    resynth = args.get("-resynth")
    uses_synthetiq = resynth == "SYNTHETIQ" or (
        resynth is None and optimization_objective in ["FT", "T"]
    )
    uses_bqskit = resynth == "BQSKIT" or (
        resynth is None and optimization_objective in ["TWO_Q", "FIDELITY"]
    )

    extra = {}
    if uses_synthetiq and "--synthetiq-binary" not in args:
        path = path_to_synthetiq or _platform_binary("main", SYNTHETIQ_BIN_DIR)
        if path is None:
            _console.print(
                "[bold red]Error:[/bold red] Unsupported platform for pre-compiled Synthetiq. "
                "Please compile Synthetiq for your platform according to "
                "https://github.com/eth-sri/synthetiq/tree/bbe3c1299a97295f5af38eec647f6bbe9fdd9234 "
                "and pass the [bold]bin/main[/bold] binary absolute path via "
                "[bold]--abs_path_to_synthetiq[/bold] or [bold]-apts[/bold]."
            )
            sys.exit(1)
        extra["--synthetiq-binary"] = os.path.abspath(path)
    if uses_bqskit and "--bqskit-worker" not in args:
        try:
            from guoq import find_bqskit_worker

            extra["--bqskit-worker"] = find_bqskit_worker()
        except (ImportError, FileNotFoundError):
            _console.print(
                "[bold red]Error:[/bold red] BQSKit resynthesis needs the worker script "
                "bundled with the [bold]guoq[/bold] pip package. Install it with "
                "[bold]pip install guoq[/bold], or pass [bold]--bqskit-worker[/bold] "
                "via advanced args."
            )
            sys.exit(1)
        extra["--python"] = sys.executable
    return extra


def print_help():
    proc = subprocess.Popen([find_guoq_binary(), "--help"])
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
        proc = subprocess.Popen(
            [find_guoq_binary(), f"@{args_file_path}"],
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
    if target_gateset in SYMB_RULES:
        args["-sr"] = os.path.join(RULES_DIR, SYMB_RULES[target_gateset])
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

    # Backend paths are filled in after the merge so advanced args can override both
    # `-resynth` and the paths themselves.
    for k, v in backend_args(args, optimization_objective, path_to_synthetiq).items():
        args.setdefault(k, v)

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

        signal.signal(signal.SIGINT, old_handler)

