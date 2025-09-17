import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser(description="Run SO(N) experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
parser.add_argument("--N", type=int, default=5, help="Dimension of SO(N)")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of random matrices")
args = parser.parse_args()

def random_so_n(n):
    # Generate a random matrix from the Haar measure on SO(n)
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    # Ensure determinant is +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

if __name__ == "__main__":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    N = args.N
    num_samples = args.num_samples

    # Generate random SO(N) matrices and compute statistics
    traces = []
    determinants = []
    for _ in range(num_samples):
        M = random_so_n(N)
        traces.append(np.trace(M))
        determinants.append(np.linalg.det(M))

    means = {
        "trace_mean": float(np.mean(traces)),
        "trace_std": float(np.std(traces)),
        "determinant_mean": float(np.mean(determinants)),
        "determinant_std": float(np.std(determinants)),
    }

    final_info = {
        "SO(N)": {
            "means": means,
            "traces": traces,
            "determinants": determinants,
        }
    }
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)
