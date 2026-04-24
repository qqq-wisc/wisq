"""Regression for issue #23: SCMR rejects unsupported gates before mapping (phased_graph)."""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from wisq import map_and_route  # noqa: E402


def test_map_and_route_rejects_rz_before_mapping(tmp_path):
    qasm = tmp_path / "circ.qasm"
    qasm.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        "qreg q[2];\nrz(0.5) q[0];\ncx q[0],q[1];\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    with pytest.raises(ValueError, match="unsupported gates"):
        map_and_route(
            str(qasm),
            "compact_layout",
            str(out),
            timeout=10,
            mode="dascot",
        )


def test_map_and_route_error_lists_unsupported_gate_names(tmp_path):
    qasm = tmp_path / "circ.qasm"
    qasm.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        "qreg q[1];\ny q[0];\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    with pytest.raises(ValueError) as excinfo:
        map_and_route(
            str(qasm),
            "compact_layout",
            str(out),
            timeout=10,
            mode="dascot",
        )
    assert "'y'" in str(excinfo.value)
