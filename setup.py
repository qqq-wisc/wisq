import subprocess
from setuptools import setup
from setuptools_rust import Binding, RustExtension

# This will run before build
missing_deps = []

# Check for gcc
try:
    subprocess.check_output(["gcc", "--version"])
except (subprocess.CalledProcessError, FileNotFoundError):
    missing_deps.append("gcc")

try:
    cmd = "java --version | head -1 | cut -f2 -d' '"
    output = subprocess.check_output([cmd], shell=True).decode()
    if int(output.split(".")[0]) < 21:
        missing_deps.append("java >= 21")
except (subprocess.CalledProcessError, FileNotFoundError):
    missing_deps.append("java >= 21")

if missing_deps:
    msg = (
        "Missing required system dependencies: {}\n"
        "Please install these dependencies before proceeding.\n"
    ).format(", ".join(missing_deps))
    raise RuntimeError(msg)

# Builds dascot_rs module
setup(
    rust_extensions=[
        RustExtension(
            "dascot_rs",
            path="Cargo.toml",
            binding=Binding.PyO3,
        )
    ],
)
