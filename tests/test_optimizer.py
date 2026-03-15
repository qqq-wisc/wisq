import pytest
import subprocess
import shutil
from utils import build_random_qasm, is_circuit_equiv
from qiskit.qasm2 import load as load_qasm_2, LEGACY_CUSTOM_INSTRUCTIONS
from pathlib import Path
import time
import random
from wisq.guoq import GATE_SETS
import csv


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

  input_name = f"input{args_dict["test_id"]}"
  input_path = Path(f"{input_name}.qasm")
  output_path = Path(f"output{args_dict["test_id"]}.qasm")

  try:
    build_random_qasm(num_qubits=num_qubits, depth=depth, seed=seed, basis_gates=gateset, dir_name=".", file_name=input_name)

    # sanity check
    assert input_path.exists(), "failed to generate QASM file"

    # build args list
    args = ["wisq", "--mode", "opt", "--output_path", output_path.as_posix(), input_path.as_posix()]  # hardcoded params needed for testing
    for arg, value in args_dict.items():
      print(value)
      if arg == "test_id":  # test_id is not input to optimizer
        continue
      elif value is None or value.lower() == "false" or value == "":  # check for blank args
        continue
      elif value.lower() == "true":  # check for flags
        args += [f"--{arg}"]
      else:
        args += [f"--{arg}", value]

    
    result = subprocess.run(
      args,
      capture_output=True,
      text=True
    )

    if result.returncode != 0:
      fail_dir = Path("failed_tests")
      fail_dir.mkdir(exist_ok=True)
      name_header = f"{gateset_name}_{num_qubits}q_{depth}d_{seed}s"
      shutil.copy(input_path, fail_dir / f"input_{name_header}.qasm")
      shutil.copy(output_path, fail_dir / f"output_{name_header}.qasm")
      pytest.fail(
        f"'wisq' failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
      )

    time.sleep(3)
    assert output_path.exists(), "optimizer did not produce output file"

    circ_in = load_qasm_2(input_path.as_posix(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    circ_out = load_qasm_2(output_path.as_posix(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)

    if not is_circuit_equiv(circ_in, circ_out):
      fail_dir = Path("failed_tests")
      fail_dir.mkdir(exist_ok=True)
      name_header = f"{gateset_name}_{num_qubits}q_{depth}d_{seed}s"
      shutil.copy(input_path, fail_dir / f"input_{name_header}.qasm")
      shutil.copy(output_path, fail_dir / f"output_{name_header}.qasm")
      pytest.fail(
        f"circuits not equivalent for {gateset_name} gateset, {num_qubits} qubits, {depth} depth\n"
        f"copied input/output file to {fail_dir}/input_{name_header}.qasm "
        f"and {fail_dir}/output_{name_header}.qasm"
      )
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
  tests the optimizer using arguments from args_limited.csv
  uses fixed num_qubits & depth (for now)
  """
  print(csv_row)
  optimizer_cli_equivalence_test(
    args_dict=csv_row
  )