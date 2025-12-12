# wisq
**wisq** is a powerful and flexible compiler for quantum circuits. It is especially well-suited for targeting fault-tolerant devices using the *surface code*. The ``full_ft`` mode optimizes the input circuit, then maps the circuit qubits to the architecture and routes two-qubit gates (including distillation-based T gates). wisq also provides a dedicated circuit optimization mode ``opt`` that can optimize circuits over [arbitrary gate sets](https://github.com/qqq-wisc/guoq?tab=readme-ov-file#supported-gate-sets) by invoking GUOQ [1] with [different arguments](#example-3-advanced-optimization-configuration).


# Dependencies
wisq requires **Python** 3.8, **Java** 21 or later, and **gcc**.


# Installation

Once the above requirements are satisfied, the ``wisq`` command line tool can be painlessly installed via pip:
```
pip install wisq
``` 

To test the installation, try the example command:
```
wisq wisq-circuits/3_17_13.qasm -ap 1e-10 -ot 10
```
If everything is working properly, the tool should run for about 10 seconds before outputting a compiled result into the file ``out.json``. (See [below](#mapping-and-routing-output-format) for how to interpret this output)


# Developer Installation 
To extend or modify wisq, you can clone the Github repo and build from the package from source. For example, using [uv](https://github.com/astral-sh/uv) for package management, this might look like:

```
git clone https://github.com/qqq-wisc/wisq.git
cd wisq
uv venv
source .venv/bin/activate
uv pip install build
python -m build --sdist
uv pip install -e .
cd src/wisq
   ```

To extend or modify the circuit optimization (GUOQ [1]/QUESO [3]) component of wisq, you will need to clone the [GUOQ repository](https://github.com/qqq-wisc/guoq), make your changes, build, and copy the new JAR to ``lib``. The Python component of GUOQ can be directly modified here in ``src/wisq/resynth.py``.
For example, the circuit optimization phase of wisq can be extended to handle new gate sets in this manner.

# Usage

## Compiler modes
wisq takes circuits in the OpenQASM 2 format as input. The compiler passes that are applied, and consequently the final output, depends on the compiler mode. This is configured with the ``--mode`` flag. The three modes are 

- ``opt``: Optimize the input circuit and write the result to a QASM file
- ``scmr``: Apply a mapping and routing pass only and output a schedule to a JSON file. 
- ``full_ft`` (default): The composition of the above; optimize the input circuit, then apply mapping and routing to the result.

The table below summarizes the compiler modes.

| Flag                                    | Description                         | Input    | Output|
| --------                                | -------                             | -------  | ----- |
| `--mode opt`                            | optimization only                   | QASM     |  QASM |
| `--mode scmr`                           | mapping/routing only                | QASM     | JSON  |
| `--mode full_ft` (or no mode specified) | optimization + mapping/routing        | QASM     | JSON  |

## Mapping and Routing Output Format
In modes that apply mapping and routing, the resulting JSON object has four keys: "map", representing the qubit map; "steps", which is a list of time steps; "arch", representing the architecture of the hardware; and "gates", which is a list of the gates from the circuit. Each step is a list of parallel gates and the paths along which they are routed. 

## Example commands
wisq includes an array of additional configuration options which can be viewed with the `wisq --help` command. Below we provide a few examples to highlight some of these options. 

### Example 1: Basic optimization configuration

Let's revisit the installation test command.


```
wisq wisq-circuits/3_17_13.qasm -ap 1e-10 -ot 10
```

Here, we run the default `full_ft` compiler mode with some configuration of the optimization. We set an approximation distance
of 10<sup>-10</sup> with the `-ap` flag and a timeout for the optimization pass with the `-ot` flag.

### Example 2: Basic mapping and routing configuration
We can target a compact architecture with less routing space using the ``-arch`` flag (see also [Custom Architectures](#Custom-Architectures))

```
wisq wisq-circuits/3_17_13.qasm --mode scmr -arch compact_layout
```

### Example 3: Advanced optimization configuration
The optimization pass can also be configured with different optimization objectives and gate sets. 

```
wisq wisq-circuits/3_17_13.qasm --mode opt -obj TOTAL -tg IBM -aa advanced_args.json
```

Here we set the optimization objective to minimize total gate with the `obj` flag and set the target gate set to the native gates on IBM machines with the `-tg` flag. 

Additionally, we use the `-aa` flag and the file ``advanced_args.json`` to pass more advanced arguments to the optimizer. The possible entries in one of these advanced arguments files can be viewed with the command `wisq --guoq-help`.

## Custom Architectures
wisq allows users to specify a custom architecture for mapping and routing. The
format for specifying an architecture is
```
{"height" : GRID_HEIGHT, "width" : GRID_WIDTH, "alg_qubits" : ALG_INDEX_LIST, "magic_states" : MS_INDEX_LIST}
```
where 
- GRID_HEIGHT and GRID_WIDTH are integers;
- ALG_INDEX_LIST and MS_VERTEX_LIST
are lists of integers indicating which grid positions are available for algorithmic qubits and which are reserved for magic states. 

To pass in a custom architecture, use the flag ``-arch PATH`` where PATH is the path to a file in the above format. 


# Benchmarks
A few example circuits are included in the ``circuits`` directory. Additional benchmarks
can be found at [this repo](https://github.com/qqq-wisc/quantum-compiler-benchmark-circuits).

# References 
wisq implements the techniques proposed in the following papers:

[1] Amanda Xu, Abtin Molavi, Swamit Tannu, Aws Albarghouthi. "[Optimizing Quantum Circuits, Fast and Slow](https://arxiv.org/abs/2411.04104)," International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS),2025


[2] Abtin Molavi, Amanda Xu, Swamit Tannu, Aws Albarghouthi. "[Dependency-Aware Compilation for Surface Code Quantum Architectures](https://arxiv.org/abs/2311.18042)" 


[3] Amanda Xu, Abtin Molavi, Lauren Pick, Swamit Tannu, Aws Albarghouthi. Synthesizing quantum-circuit optimizers. Proceedings of the ACM on Programming Languages. Volume 7, PLDI, 2023. https://doi.org/10.1145/3591254
