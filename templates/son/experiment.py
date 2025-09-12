import argparse
import json
import os
from typing import Tuple

import numpy as np


def generate_so_element(n: int, complexified: bool = False) -> np.ndarray:
	xi = np.random.normal(0.0, 1.0, size=(n, n))
	if complexified:
		eta = np.random.normal(0.0, 1.0, size=(n, n))
		return 0.5 * (xi - xi.T) + 0.5j * (eta - eta.T)
	return 0.5 * (xi - xi.T)


def random_group_element_from_algebra(X: np.ndarray, step: float = 1.0) -> np.ndarray:
	"""
	so(n) は実直交群 SO(n) のリー代数。指数写像 exp: so(n) -> SO(n) を用いる。
	X が反対称なら exp(step * X) は直交かつ det=+1 になる（十分小さな step では det>0 を保ちやすい）。
	"""
	from scipy.linalg import expm
	G = expm(step * X)
	# 数値誤差対策として直交性を強制投影（極分解）してもよいが、ここでは簡潔に。
	return G


def sample_dataset(n: int, m: int, complexified: bool = False, seed: int = 1337) -> Tuple[np.ndarray, np.ndarray]:
	"""
	m サンプル分、ランダムに so(n) の元 X を生成し、G = exp(X) を対として返す。
	戻り値:
	  Xs: 形状 (m, n, n)
	  Gs: 形状 (m, n, n)
	"""
	rng = np.random.default_rng(seed)
	Xs = []
	Gs = []
	for _ in range(m):
		# 分散スケールは 1 を基本とし、安定のため step を小さめに
		X = generate_so_element(n, complexified=complexified)
		G = random_group_element_from_algebra(X, step=1.0)
		Xs.append(X)
		Gs.append(G)
	return np.stack(Xs), np.stack(Gs)


def main():
	parser = argparse.ArgumentParser(description="SO(n) random generation experiment")
	parser.add_argument("--n", type=int, default=5, help="Dimension n for SO(n)")
	parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
	parser.add_argument("--complexified", action="store_true", help="Use complexified algebra so(n)_C")
	parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
	parser.add_argument("--seed", type=int, default=1337, help="Random seed")
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	Xs, Gs = sample_dataset(args.n, args.num_samples, complexified=args.complexified, seed=args.seed)

	# 基本的な検証: 直交性 G^T G = I を平均誤差で確認
	I = np.eye(args.n)
	orth_errs = []
	for G in Gs:
		orth_errs.append(np.linalg.norm(G.T @ G - I) / np.linalg.norm(I))
	orth_err = float(np.mean(orth_errs))

	final_info = {
		"n": args.n,
		"num_samples": args.num_samples,
		"complexified": bool(args.complexified),
		"orthogonality_error_mean": orth_err,
	}
	print(final_info)
	with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
		json.dump(final_info, f)

	# 保存（必要に応じて）
	np.save(os.path.join(args.out_dir, "Xs.npy"), Xs)
	np.save(os.path.join(args.out_dir, "Gs.npy"), Gs)


if __name__ == "__main__":
	main()


