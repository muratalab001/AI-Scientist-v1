import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
def load_result(out_dir):
    with open(os.path.join(out_dir, "final_info.json"), encoding="utf-8") as f:
        return json.load(f)

def plot_eigvals(result, out_dir):
    eigvals = result["eigvals"]
    N = result["N"]
    plt.figure(figsize=(8, 6))
    for eig in eigvals:
        eig = np.array(eig)
        plt.scatter(eig.real, eig.imag, alpha=0.7)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title(f"Eigenvalue spectra of SO({N}) matrices")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"so{N}_eigvals.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    args = parser.parse_args()
    result = load_result(args.out_dir)
    plot_eigvals(result, args.out_dir)
