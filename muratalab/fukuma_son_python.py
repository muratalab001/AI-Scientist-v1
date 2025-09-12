import numpy as np


def transpose(X):
	return X.T


def ndist_matrix(nc):
	# 平均0, 分散1の正規分布に従う実数値行列
	return np.random.normal(0.0, 1.0, size=(nc, nc))


def generate_lie_so(nc):
	"""
	so(n) のランダム元を生成（実反対称行列）。
	X = (xi - xi^T)/2 により X^T = -X を満たす。
	"""
	xi = ndist_matrix(nc)
	X = 0.5 * (xi - transpose(xi))
	return X


def generate_lie_so_complex(nc):
	"""
	so(n)_C のランダム元を生成（複素反対称行列）。
	Z = (xi - xi^T)/2 + i*(eta - eta^T)/2。
	"""
	xi = ndist_matrix(nc)
	eta = ndist_matrix(nc)
	Z = 0.5 * (xi - transpose(xi)) + 0.5j * (eta - transpose(eta))
	return Z


