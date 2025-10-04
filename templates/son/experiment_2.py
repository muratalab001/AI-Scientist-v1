import argparse
import json
import os
from typing import List, Tuple

import numpy as np


def pair_indices(n: int) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def so_coords_from_matrix(X: np.ndarray) -> np.ndarray:
    """
    Project an antisymmetric matrix X onto the orthonormal basis
    T^{(ij)} = (E_{ij} - E_{ji})/sqrt(2), i<j, such that tr(T^a T^b) = -delta^{ab}.

    Returns the coordinate vector x with x^{(ij)} = sqrt(2) * X_{ij} (for i<j).
    """
    n = X.shape[0]
    coords = []
    for i in range(n):
        for j in range(i + 1, n):
            coords.append(np.sqrt(2.0) * X[i, j])
    return np.asarray(coords)


def random_so_algebra(n: int, sigma: float, complexified: bool = False):
    """
    Sample a random element from the Gaussian on so(n) or so(n,C) without
    using an explicit basis, following the SU(n) construction adapted to SO(n):

    Real case (so(n)):
      X = 1/2 * (xi - xi^T), with xi_{ij} ~ N(0, sigma^2) i.i.d.

    Complexified case (so(n,C)):
      Z = 1/2 * [ (xi - xi^T) + i (eta - eta^T) ], with xi,eta i.i.d. N(0, sigma^2).
    """
    if not complexified:
        xi = np.random.normal(loc=0.0, scale=sigma, size=(n, n))
        X = 0.5 * (xi - xi.T)
        return X
    else:
        xi = np.random.normal(loc=0.0, scale=sigma, size=(n, n))
        eta = np.random.normal(loc=0.0, scale=sigma, size=(n, n))
        Z = 0.5 * ((xi - xi.T) + 1j * (eta - eta.T))
        return Z


def main():
    parser = argparse.ArgumentParser(description="Gaussian sampling on so(N) (Fukuma SU(N)->SO(N) adaptation)")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument("--N", type=int, default=5, help="Dimension n for so(n)")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of random algebra elements")
    parser.add_argument("--sigma", type=float, default=1.0, help="Stddev for Gaussian on coordinates x^a")
    parser.add_argument("--complex", dest="complexified", action="store_true", help="Sample in so(n,C) instead of so(n)")
    parser.add_argument("--no-complex", dest="complexified", action="store_false")
    parser.set_defaults(complexified=False)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    n = args.N
    num_samples = args.num_samples
    sigma = args.sigma
    complexified = args.complexified

    # Accumulators for summary statistics
    coords_sum = None
    coords_sq_sum = None
    entry_mean = np.zeros((n, n), dtype=np.complex128 if complexified else np.float64)
    norm2_list = []  # -tr(X^2) for real, tr(Z*Z^\dagger) for complex

    # Precompute pair ordering to give stable coordinate outputs
    pairs = pair_indices(n)
    dimG = len(pairs)  # n(n-1)/2

    for _ in range(num_samples):
        A = random_so_algebra(n, sigma, complexified)

        if complexified:
            # so(n,C): Z^T = -Z
            assert np.allclose(A.T, -A), "Sampled Z is not antisymmetric"
            # Frobenius norm squared equals sum_a |z^a|^2 under our normalization
            norm2 = float(np.real(np.vdot(A, A)))  # tr(A^* A)
            norm2_list.append(norm2)

            # Only accumulate coordinate stats for the real/imag parts separately if needed.
            # Here we collect coordinates of Re(Z) and Im(Z) concatenated for summary.
            Re = A.real
            Im = A.imag
            x = so_coords_from_matrix(Re)
            y = so_coords_from_matrix(Im)
            vec = np.concatenate([x, y])  # length 2*dimG
            if coords_sum is None:
                coords_sum = np.zeros_like(vec)
                coords_sq_sum = np.zeros_like(vec)
            coords_sum += vec
            coords_sq_sum += vec * vec

            entry_mean += A
        else:
            # so(n): X^T = -X
            assert np.allclose(A.T, -A), "Sampled X is not antisymmetric"
            # -tr(X^2) equals sum_a (x^a)^2 under our normalization
            norm2 = float(-np.trace(A @ A))
            norm2_list.append(norm2)

            x = so_coords_from_matrix(A)
            if coords_sum is None:
                coords_sum = np.zeros_like(x)
                coords_sq_sum = np.zeros_like(x)
            coords_sum += x
            coords_sq_sum += x * x

            entry_mean += A

    entry_mean /= num_samples

    if complexified:
        # For complex case we reported concatenated coords (x then y)
        coords_mean = (coords_sum / num_samples).tolist()
        coords_std = (np.sqrt(np.maximum(0.0, coords_sq_sum / num_samples - (coords_sum / num_samples) ** 2))).tolist()
        theory_norm2_mean = float(2.0 * dimG * sigma * sigma)
    else:
        coords_mean = (coords_sum / num_samples).tolist()
        coords_std = (np.sqrt(np.maximum(0.0, coords_sq_sum / num_samples - (coords_sum / num_samples) ** 2))).tolist()
        theory_norm2_mean = float(dimG * sigma * sigma)

    means = {
        "coords_mean": coords_mean,  # expected ~0
        "coords_std": coords_std,    # expected ~sigma (real) or [sigma, sigma] (complex parts)
        "norm2_mean": float(np.mean(norm2_list)),
        "norm2_std": float(np.std(norm2_list)),
        "theory_norm2_mean": theory_norm2_mean,
    }

    final_info = {
        "SO(N)_LieAlg_Gaussian": {
            "n": n,
            "dimG": dimG,
            "num_samples": num_samples,
            "sigma": sigma,
            "complexified": complexified,
            "means": means,
            # For small n this is handy to quickly inspect symmetry and bias
            "entry_mean": entry_mean.real.tolist() if not complexified else {
                "real": entry_mean.real.tolist(),
                "imag": entry_mean.imag.tolist(),
            },
        }
    }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)


if __name__ == "__main__":
    main()

