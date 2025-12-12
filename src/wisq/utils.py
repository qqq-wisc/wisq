import os
from time import time_ns
import random


def create_scratch_dir(output_path: str) -> str:
    # Create temporary scratch directory for GUOQ
    timestamp = time_ns()
    uid = f"{timestamp}_{random.randint(0, 10000000)}"
    scratch_dir_name = f"wisq_tmp_{uid}"
    scratch_dir_path = os.path.join(os.path.dirname(output_path), scratch_dir_name)
    os.mkdir(scratch_dir_path)
    return (scratch_dir_path, uid)
