import argparse
import json
import os
import numpy as np
from scipy.stats import ortho_group

def random_so_n(n):
    # Generate an orthogonal matrix with det=1 (SO(N))
    Q = ortho_group.rvs(dim=n)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

parser = argparse.ArgumentParser(description="Numerical experiments for SO(N) gauge group")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
parser.add_argument("--N", type=int, default=3, help="N for SO(N)")
args = parser.parse_args()

if __name__ == "__main__":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    N = args.N

    # Generate samples of SO(N) matrices
    so_n_matrices = [random_so_n(N) for _ in range(10)]

    # Example: Calculate eigenvalues of each matrix
    eigvals = [np.linalg.eigvals(M).tolist() for M in so_n_matrices]

    # Save results
    result = {
        "N": N,
        "eigvals": eigvals,
        "description": "Eigenvalue spectra of 10 SO(N) matrices"
    }
    with open(os.path.join(out_dir, "final_info.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
