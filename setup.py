import subprocess
from setuptools import setup

# This will run before build
missing_deps = []

# Check for gcc
try:
    subprocess.check_output(["gcc", "--version"])
except (subprocess.CalledProcessError, FileNotFoundError):
    missing_deps.append("gcc")

if missing_deps:
    msg = (
        "Missing required system dependencies: {}\n"
        "Please install these dependencies before proceeding.\n"
    ).format(", ".join(missing_deps))
    raise RuntimeError(msg)

setup()
