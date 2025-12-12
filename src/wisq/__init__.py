import argparse
import ast
from qiskit import QuantumCircuit
from .architecture import square_sparse_layout, compact_layout
from .dascot import (
    extract_gates_from_file,
    extract_qubits_from_gates,
    dump,
    run_dascot,
    run_sat_scmr,
)
from .guoq import run_guoq, print_help, CLIFFORDT, FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE
from .utils import create_scratch_dir
import os
import shutil
import json


OPT_MODE = "opt"
FULL_FT_MODE = "full_ft"
SCMR_MODE = "scmr"

DEFAULT_EXT = {
    OPT_MODE: "qasm",
    FULL_FT_MODE: "json",
    SCMR_MODE: "json",
}


class Guoq_Help_Action(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help=None,
    ):
        super(Guoq_Help_Action, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print_help()
        parser.exit()


def map_and_route(
    input_path: str, arch_name: str, output_path: str, timeout: int, mode="dascot"
):
    """
    Apply a surface code mapping and routing pass to the given circuit

    Args:
        input_path: Path to the input circuit file.
        output_path: Path to the output circuit file.
        arch_name: A description of the target architecture. Valid options are the
        strings "square_sparse_layout" or "compact_layout" representing built-in architectures,
        or the path to a text file containing the description of a custom architecture
        timeout: Total timeout in seconds for both mapping and routing.


    Writes a JSON representing
    the scheduled circuit after mapping and routing to output_path.
    """
    circ = QuantumCircuit.from_qasm_file(input_path)
    gates, ops = extract_gates_from_file(input_path)
    id_to_op = {i: ops[i] for i in range(len(ops))}
    total_qubits = len(extract_qubits_from_gates(gates))
    circ = QuantumCircuit.from_qasm_file(input_path)
    if arch_name == "square_sparse_layout":
        arch = square_sparse_layout(total_qubits, magic_states="all_sides")
    elif arch_name == "compact_layout":
        arch = compact_layout(total_qubits, magic_states="all_sides")
    else:
        with open(arch_name) as f:
            arch = ast.literal_eval(f.read())
    if mode == "dascot":
        map, steps = run_dascot(circ, gates, arch, output_path, timeout)
    elif mode == "sat":
        map, steps = run_sat_scmr(circ, gates, arch, output_path, timeout)
    dump(arch, map, steps, id_to_op, output_path, gates)


def optimize(
    input_path: str,
    output_path: str,
    target_gateset: str,
    optimization_objective: str,
    timeout: int,
    approximation_epsilon: float = 0,
    advanced_args: dict = None,
    verbose: bool = False,
    path_to_synthetiq: str = None,
) -> None:
    """
    Use the default GUOQ parameters to optimize a circuit. Recommended for most users. Advanced users can use `advanced_args` to override default values.

    Args:
        input_path: Path to the input circuit file.
        output_path: Path to the output circuit file.
        target_gateset: Target gateset to optimize the circuit to.
        timeout: Timeout in seconds. If set to 0, only transpiles and performs no optimization.
        approximation_epsilon: Approximation epsilon to use.
        advanced_args: Dictionary containing advanced arguments to pass to GUOQ, overriding default values except `-out` and `-job`. `guoq.print_help` displays available options.
        For example, if we want to override the default for `--rules` and use the `--remove-size-preserving-rules` flag, the dictionary would be `{"--rules": "file.txt", "--remove-size-preserving-rules": None}`.
        verbose: Whether to print verbose output.
    """
    run_guoq(
        input_path,
        output_path,
        target_gateset,
        optimization_objective,
        timeout,
        approximation_epsilon=approximation_epsilon,
        args=advanced_args,
        verbose=verbose,
        path_to_synthetiq=path_to_synthetiq,
    )


def compile_fault_tolerant(
    input_path,
    output_path,
    opt_timeout,
    arch_name,
    approximation_epsilon=1e-10,
    verbose=False,
    mr_timeout=1800,
    mr_solver="dascot",
    path_to_synthetiq=None,
):
    """
    Compiles a circuit to a fault-tolerant architecture using the Clifford + T gate set.
    The input is a QASM circuit and architecture and the output is a JSON representing
    the scheduled circuit after optimizing, mapping, and routing.
    """
    scratch_dir_path, _ = create_scratch_dir(output_path)

    try:
        transpiled_and_optimized_path = os.path.join(
            scratch_dir_path, "after_guoq.qasm"
        )
        print(
            f"Decomposing to Clifford + T (if needed) and optimizing the input circuit with a timeout of {opt_timeout} seconds..."
        )
        optimize(
            input_path,
            transpiled_and_optimized_path,
            CLIFFORDT,
            FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE,
            opt_timeout,
            approximation_epsilon,
            verbose=verbose,
            path_to_synthetiq=path_to_synthetiq,
        )
        print(
            f"Done optimizing. Mapping and routing the optimized circuit with a timeout of {mr_timeout} seconds..."
        )
        map_and_route(
            transpiled_and_optimized_path,
            arch_name,
            output_path,
            mr_timeout,
            mode=mr_solver,
        )
    finally:
        if os.path.exists(scratch_dir_path):
            shutil.rmtree(scratch_dir_path)


def main():
    parser = argparse.ArgumentParser(
        prog="wisq",
        description="A compiler for quantum circuits. Optimize your circuits and/or map them to a surface code architecture. See README for example usage and documentation.",
    )
    opt = parser.add_argument_group(title="optimization config")
    scmr = parser.add_argument_group(title="mapping and routing config")
    parser.add_argument("input_path", help="path to the input circuit")
    parser.add_argument(
        "--output_path",
        "-op",
        help="path to write the output. Default is out.qasm or out.json",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default=FULL_FT_MODE,
        help="""
        control which compilation passes to apply. |
        opt: circuit optimization only |
        scmr: surface code mapping and routing only |
        full_ft (default): circuit optimization, then surface code mapping and routing
        """,
        choices=[OPT_MODE, FULL_FT_MODE, SCMR_MODE],
    )
    opt.add_argument(
        "--target_gateset",
        "-tg",
        help="target gateset for circuit optimization (default: Clifford + T)",
        default=CLIFFORDT,
        choices=guoq.GATE_SETS.keys(),
    )
    opt.add_argument(
        "--optimization_objective",
        "-obj",
        help="objective function used to guide the optimization (default: fault-tolerant objective function)",
        default=FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE,
        choices=["TWO_Q", "FIDELITY", "FT", "TOTAL", "T"],
    )
    opt.add_argument(
        "--opt_timeout",
        "-ot",
        help="integer representing timeout for optimization in seconds (default: 3600)",
        type=int,
        default=3600,
    )
    opt.add_argument(
        "--approx_epsilon",
        "-ap",
        help="the approximation epsilon for optimization (represented as a plain decimal or in scientific notation, e.g., 1e-8); output of optimization pass is equivalent to input circuit up to epsilon error (default: 0)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--verbose", "-v", help="print verbose output", action="store_true"
    )
    scmr.add_argument(
        "--architecture",
        "-arch",
        help="target architecture for mapping and routing, can be one of the strings {'square_sparse_layout', 'compact_layout'} or the path to a file specifying a custom architecture. See README for format. (default: 'square_sparse')",
        default="square_sparse_layout",
    )
    scmr.add_argument(
        "--mr_timeout",
        "-tmr",
        type=int,
        help="integer representing timeout for mapping and routing in seconds (default: 1800)",
        default=1800,
    )
    scmr.add_argument(
        "--mr_solver",
        "-smr",
        help="solver to use for mapping and routing (default: 'dascot')",
        default="dascot",
    )
    parser.add_argument(
        "--guoq_help", "-gh", help="print GUOQ options", action=Guoq_Help_Action
    )
    parser.add_argument(
        "--advanced_args",
        "-aa",
        help="file path to JSON with advanced GUOQ args. See `optimize` for example",
    )
    parser.add_argument(
        "--abs_path_to_synthetiq",
        "-apts",
        help="absolute path to Synthetiq `main` binary",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_path) and "wisq-circuits" in args.input_path:
        args.input_path = os.path.join(
            os.path.dirname(__file__),
            args.input_path,
        )

    if args.output_path is None:
        args.output_path = f"out.{DEFAULT_EXT[args.mode]}"

    if args.advanced_args is not None:
        with open(args.advanced_args, "r") as f:
            args.advanced_args = json.load(f)

    if args.mode == OPT_MODE:
        optimize(
            input_path=args.input_path,
            output_path=args.output_path,
            target_gateset=args.target_gateset,
            optimization_objective=args.optimization_objective,
            timeout=args.opt_timeout,
            approximation_epsilon=args.approx_epsilon,
            advanced_args=args.advanced_args,
            verbose=args.verbose,
            path_to_synthetiq=args.abs_path_to_synthetiq,
        )
    elif args.mode == FULL_FT_MODE:
        compile_fault_tolerant(
            input_path=args.input_path,
            output_path=args.output_path,
            opt_timeout=args.opt_timeout,
            arch_name=args.architecture,
            approximation_epsilon=args.approx_epsilon,
            verbose=args.verbose,
            mr_timeout=args.mr_timeout,
            mr_solver=args.mr_solver,
            path_to_synthetiq=args.abs_path_to_synthetiq,
        )
    elif args.mode == SCMR_MODE:
        map_and_route(
            input_path=args.input_path,
            output_path=args.output_path,
            arch_name=args.architecture,
            timeout=args.mr_timeout,
            mode=args.mr_solver,
        )


if __name__ == "__main__":
    main()
