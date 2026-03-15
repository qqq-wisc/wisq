import pytest
import subprocess
import shutil
from utils import build_random_qasm, is_circuit_equiv, build_random_cliffordt_qasm
from qiskit.qasm2 import load as load_qasm_2, LEGACY_CUSTOM_INSTRUCTIONS
from pathlib import Path
import time
import random
from wisq.guoq import GATE_SETS
import csv

FAIL_DIR = Path("failed_tests")

def load_csv_args(path):
  rows = []
  with open(path, newline="") as f:
    reader = csv.DictReader(f)
    counter = 0
    for row in reader:
      row["test_id"] = counter
      rows.append(row)
      counter+=1
  return rows


def handle_error_code(name_header, input_path, output_path):
  fail_dir = FAIL_DIR
  fail_dir.mkdir(exist_ok=True)
  shutil.copy(input_path, fail_dir / f"input_{name_header}.qasm")
  shutil.copy(output_path, fail_dir / f"output_{name_header}.qasm")


def build_args(mode, args_dict, output_path, input_path):
  args = ["wisq", "--mode", mode, "--output_path", output_path.as_posix()]
  for arg, value in args_dict.items():
    if arg == "test_id":  # test_id is not input to wisq
      continue
    elif value is None or value.lower() == "false" or value == "":  # check for blank args
      continue
    elif value.lower() == "true":  # check for flags
      args += [f"--{arg}"]
    else:
      args += [f"--{arg}", value]
  args += [input_path.as_posix()]
  return args


def optimizer_cli_equivalence_test(
  args_dict: dict[str, str]
):
  """
  tests the optimizer by generating a random circuit, running it
  through the optimizer, then checking equivalence

  this is done with a random gateset as well, unless specified through
  the target_gateset param
  """
  num_qubits: int = 4
  depth: int = 10

  seed = int(time.time())
  random.seed(seed)
  print(f"[INFO] Running with seed={seed}")
  
  gateset_name = args_dict["target_gateset"].upper() if args_dict["target_gateset"] else random.choice(list(GATE_SETS.keys()))
  gateset = GATE_SETS[gateset_name]

  input_name = f"opt_input{args_dict["test_id"]}"
  input_path = Path(f"{input_name}.qasm")
  output_path = Path(f"opt_output{args_dict["test_id"]}.qasm")

  try:
    build_random_qasm(num_qubits=num_qubits, depth=depth, seed=seed, basis_gates=gateset, dir_name=".", file_name=input_name)

    # sanity check
    assert input_path.exists(), "failed to generate QASM file"

    # build args list
    args = build_args(mode="opt", args_dict=args_dict, output_path=output_path, input_path=input_path)
    
    result = subprocess.run(
      args,
      capture_output=True,
      text=True
    )

    if result.returncode != 0:
      name_header = f"{gateset_name}_{num_qubits}q_{depth}d_{seed}s"
      handle_error_code(name_header, input_path, output_path)
      pytest.fail(
        f"'wisq' failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
      )

    time.sleep(3)
    assert output_path.exists(), "did not produce output file"

    circ_in = load_qasm_2(input_path.as_posix(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    circ_out = load_qasm_2(output_path.as_posix(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)

    if not is_circuit_equiv(circ_in, circ_out): # TODO: handle non-exact checks
      name_header = f"{gateset_name}_{num_qubits}q_{depth}d_{seed}s"
      handle_error_code(name_header, input_path, output_path)
      pytest.fail(
        f"circuits not equivalent for {gateset_name} gateset, {num_qubits} qubits, {depth} depth\n"
        f"copied input/output file to {FAIL_DIR}/input_{name_header}.qasm "
        f"and {FAIL_DIR}/output_{name_header}.qasm"
      )
  finally:
    for path in [input_path, output_path]:
      if path.exists():
        path.unlink()


def mr_cli_test(
  args_dict: dict[str, str]
):
  """
  tests dascot by generating a random circuit and checking if output json is created
  """
  num_qubits: int = 4
  depth: int = 4

  seed = int(time.time())
  random.seed(seed)
  print(f"[INFO] Running with seed={seed}")

  input_name = f"mr_input{args_dict["test_id"]}"
  input_path = Path(f"{input_name}.qasm")
  output_path = Path(f"mr_output{args_dict["test_id"]}.json")

  try:
    build_random_cliffordt_qasm(num_qubits=num_qubits, depth=depth, seed=seed, epsilon=1e-3, dir_name=".", file_name=input_name)

    # sanity check
    assert input_path.exists(), "failed to generate QASM file"

    # build args list
    args = build_args(mode="scmr", args_dict=args_dict, output_path=output_path, input_path=input_path)

    result = subprocess.run(
      args,
      capture_output=True,
      text=True
    )

    if result.returncode != 0:
      name_header = f"{num_qubits}q_{depth}d_{seed}s"
      handle_error_code(name_header, input_path, output_path)
      pytest.fail(
        f"'wisq' failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
      )

    time.sleep(3)
    assert output_path.exists(), "did not produce output file"

  finally:
    for path in [input_path, output_path]:
      if path.exists():
        path.unlink()


def full_ft_cli_test(
  args_dict: dict[str, str]
):
  """
  tests full_ft by generating a random circuit and checking if output json is created
  """
  num_qubits: int = 4
  depth: int = 3

  seed = int(time.time())
  random.seed(seed)
  print(f"[INFO] Running with seed={seed}")

  gateset_name = args_dict["target_gateset"].upper() if args_dict["target_gateset"] else random.choice(list(GATE_SETS.keys()))
  gateset = GATE_SETS[gateset_name]

  input_name = f"fullft_input{args_dict["test_id"]}"
  input_path = Path(f"{input_name}.qasm")
  output_path = Path(f"fullft_output{args_dict["test_id"]}.json")

  try:
    if gateset_name == "CLIFFORDT":
      build_random_cliffordt_qasm(num_qubits=num_qubits, depth=depth, seed=seed, epsilon=1e-3, dir_name=".", file_name=input_name)
    else:
      build_random_qasm(num_qubits=num_qubits, depth=depth, seed=seed, basis_gates=gateset, dir_name=".", file_name=input_name)

    # sanity check
    assert input_path.exists(), "failed to generate QASM file"

    # build args list
    args = build_args(mode="full_ft", args_dict=args_dict, output_path=output_path, input_path=input_path)
    
    result = subprocess.run(
      args,
      capture_output=True,
      text=True
    )

    if float(args_dict["approx_epsilon"]) == 0 and args_dict["target_gateset"] != "CLIFFORDT":
      assert result.returncode != 0, "should give error msg and quit"
      assert "Please pass a value strictly between 0 and 1 to" in result.stdout
      return

    if result.returncode != 0:
      name_header = f"{num_qubits}q_{depth}d_{seed}s"
      handle_error_code(name_header, input_path, output_path)
      pytest.fail(
        f"'wisq' failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
      )

    time.sleep(3)
    assert output_path.exists(), "did not produce output file"

  finally:
    for path in [input_path, output_path]:
      if path.exists():
        path.unlink()


@pytest.mark.parametrize(
  "csv_row",
  load_csv_args("opt_test_args.csv")  
)
def test_optimizer_cli_equivalence_csv(csv_row):
  """
  tests the optimizer using arguments from opt_test_args.csv
  uses fixed num_qubits & depth (for now)
  """
  print(csv_row)
  optimizer_cli_equivalence_test(
    args_dict=csv_row
  )


@pytest.mark.parametrize(
  "csv_row",
  load_csv_args("mr_test_args.csv")  
)
def test_mr_cli_csv(csv_row):
  """
  tests the mapper and router using arguments from mr_test_args.csv
  uses fixed num_qubits & depth (for now)
  """
  print(csv_row)
  mr_cli_test(
    args_dict=csv_row
  )


@pytest.mark.parametrize(
  "csv_row",
  load_csv_args("full_ft_test_args.csv")  
)
def test_full_ft_cli_csv(csv_row):
  """
  tests the full_ft mode using arguments from full_ft_test_args.csv
  uses fixed num_qubits & depth (for now)
  """
  print(csv_row)
  full_ft_cli_test(
    args_dict=csv_row
  )