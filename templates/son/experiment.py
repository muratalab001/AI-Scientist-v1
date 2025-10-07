import argparse
import json
import os

import numpy as np


def matrix_exponential(A: np.ndarray) -> np.ndarray:
    # Compute matrix exponential via eigendecomposition: exp(A) = V exp(D) V^{-1}
    vals, vecs = np.linalg.eig(A)
    expD = np.diag(np.exp(vals))
    inv_vecs = np.linalg.inv(vecs)
    return vecs @ expD @ inv_vecs


def generate_su_n_lie_algebra_element(n: int, sigma: float) -> np.ndarray:
    # Draw complex matrix with entries ~ N(0, sigma/sqrt(2)) for real/imag parts
    scale = sigma / np.sqrt(2.0)
    real_part = np.random.normal(loc=0.0, scale=scale, size=(n, n))
    imag_part = np.random.normal(loc=0.0, scale=scale, size=(n, n))
    H = real_part + 1j * imag_part
    # Anti-Hermitian: X = (H - H^†) / 2
    X = 0.5 * (H - H.conj().T)
    # Traceless: subtract (Tr(X)/n) * I
    tr = np.trace(X)
    X = X - (tr / n) * np.eye(n, dtype=complex)
    return X


def generate_su_n_element(n: int, sigma: float) -> np.ndarray:
    X = generate_su_n_lie_algebra_element(n, sigma)
    U = matrix_exponential(X)
    return U


def generate_su_n_sample(n: int, sigma: float, sample_size: int):
    return [generate_su_n_element(n, sigma) for _ in range(sample_size)]


def check_su_n(U: np.ndarray, n: int, tolerance: float = 1e-10) -> bool:
    det_check = np.abs(np.linalg.det(U) - 1.0) < tolerance
    unitary_check = np.max(np.abs(U.conj().T @ U - np.eye(n))) < tolerance
    return bool(det_check and unitary_check)


def complex_to_json_safe(x: complex) -> float:
    # For metrics we store magnitudes where complex would otherwise appear
    return float(np.abs(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SU(N) experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # SU(2) element with sigma = 1.0
    su2 = generate_su_n_element(2, sigma=1.0)
    su2_det = np.linalg.det(su2)
    su2_det_abs_diff = float(np.abs(su2_det - 1.0))
    su2_unitary_max_dev = float(np.max(np.abs(su2.conj().T @ su2 - np.eye(2))))

    # SU(3) sample of 5 elements with sigma = 0.5
    su3_sample = generate_su_n_sample(3, sigma=0.5, sample_size=5)
    su3_valid_flags = [check_su_n(U, 3) for U in su3_sample]
    su3_valid_count = int(np.sum(su3_valid_flags))
    su3_valid_fraction = float(su3_valid_count / len(su3_sample))

    # Lie algebra element properties for n=3, sigma = 1.0
    lie_el = generate_su_n_lie_algebra_element(3, sigma=1.0)
    lie_trace_abs = complex_to_json_safe(np.trace(lie_el))
    lie_antiherm_max = float(np.max(np.abs(lie_el + lie_el.conj().T)))

    # Effect of sigma on distance to identity for SU(2)
    sigmas = [0.1, 0.5, 1.0, 2.0]
    sigma_effect = {}
    for s in sigmas:
        U = generate_su_n_element(2, sigma=s)
        norm_frob = float(np.linalg.norm(U - np.eye(2), ord="fro"))
        sigma_effect[str(s)] = norm_frob

    means = {
        "su2_det_abs_diff": su2_det_abs_diff,
        "su2_unitary_max_deviation": su2_unitary_max_dev,
        "lie_trace_abs": lie_trace_abs,
        "lie_antihermitian_max": lie_antiherm_max,
        "su3_valid_fraction": su3_valid_fraction,
        # Include sigma norms as separate scalars for comparability across runs
        **{f"sigma_{k}_norm": v for k, v in sigma_effect.items()},
    }

    details = {
        "su2_det": [float(np.real(su2_det)), float(np.imag(su2_det))],
        "su2_unitary_max_deviation": su2_unitary_max_dev,
        "su3_valid_flags": su3_valid_flags,
        "sigma_effect_norms": sigma_effect,
    }

    final_info = {
        "SU(N)": {
            "means": means,
            "details": details,
        }
    }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)

