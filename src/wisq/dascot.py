import json
import re
import time
from .sarouting import TimeoutException
from .phased_graph import build_phased_map
from .sarouting import sim_anneal_route
from .sat_scmr import solve
import signal
from rich.console import Console

_console = Console()


def extract_gates_from_file(fname):
    gates = []
    ops = []
    with open(fname) as f:
        for line in f:
            match = re.match(r"(cx)\s+q\[(\d+)\],\s*q\[(\d+)\];", line)
            if match:
                gates.append((int(match.group(2)), int(match.group(3))))
                ops.append(match.group(1))
            match = re.match(r"(t|tdg)\s+q\[(\d+)\];", line)
            if match:
                ops.append(match.group(1))
                gates.append((int(match.group(2)),))
    return gates, ops


def extract_qubits_from_gates(gate_list):
    qubits = set()
    for gate in gate_list:
        for qubit in gate:
            qubits.add(qubit)
    return qubits


def dump(arch, map, steps, id_to_op, output_path, gates, interrupted=False):
    output = {}
    output["map"] = {k: v for k, v in map}
    output["steps"] = [label_step(id_to_op, step) for step in steps]
    output["arch"] = arch
    output["gates"] = gates
    output["fully_routed?"] = not interrupted
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)


def label_step(id_to_op, step):
    return [labeled_gate_path(id_to_op, *gate_path) for gate_path in step]


def labeled_gate_path(id_to_op, id, args, path):
    dict = {}
    dict["id"] = id
    dict["op"] = id_to_op[id]
    dict["qubits"] = args
    dict["path"] = path
    return dict


def run_dascot(circ, gates, arch, output_path, timeout):
    start = time.time()
    sim_anneal_params = [100, 0.1, 0.1]
    depth = circ.depth(filter_function=lambda x: x[0].name in ["cx", "t", "tdg"])
    scaled_sim_anneal_params = [
        sim_anneal_params[0],
        sim_anneal_params[1] / depth,
        10 * sim_anneal_params[2] / depth,
    ]
    phased_map, _ = build_phased_map(
        extract_qubits_from_gates(gates),
        circ,
        arch,
        include_t=True,
        timeout=timeout // 2,
        *scaled_sim_anneal_params,
    )
    map_end = time.time()
    elapsed = map_end - start
    # alarm required an integer >= 1
    remaining_timeout = max(1, timeout - elapsed)
    steps, _, interrupted = sim_anneal_route(
        gates,
        arch,
        phased_map,
        reward_name="criticality",
        order_fraction=1,
        take_first_ms=False,
        timeout=remaining_timeout,
        *[10, 0.1, 0.1],
    )
    if interrupted:
        _console.print(
            "    [bold yellow]⚠[/bold yellow]  Routing timed out — writing partial output"
        )
    return phased_map, steps, interrupted


def run_sat_scmr(circ, gates, arch, output_path, timeout):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    depth = circ.depth(filter_function=lambda x: x[0].name in ["cx", "t", "tdg"])
    width = arch["width"]
    height = arch["height"]
    msf_faces = arch["magic_states"]
    alg_qubits = arch["alg_qubits"]
    try:
        map, steps = solve(
            gates=gates,
            msf_faces=msf_faces,
            grid_len=width,
            grid_height=height,
            alg_qubits=alg_qubits,
            start_from=depth,
        )
    except TimeoutException:
        _console.print(
            "    [bold yellow]⚠[/bold yellow]  Mapping and routing timed out - exiting with empty solution. Consider increasing the timeout or using the DASCOT solver.",
        )
        return {}, []
    finally:
        signal.alarm(0)
    return map, steps
