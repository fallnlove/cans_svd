import pandas as pd
import os
import time
from tqdm import tqdm
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, List, Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.lax.linalg import SvdAlgorithm

from src_jax.svd import svd


@dataclass
class AlgorithmSpec:
    name: str
    precision: str
    fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]


def compute_error(A, U, s,  Vt):
    precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "float32")
    recon = jnp.linalg.norm(A - U @ (jnp.diag(s) @ Vt)) / jnp.linalg.norm(A)
    u_ortho = jnp.linalg.norm(U.T.conj() @ U - jnp.eye(U.shape[1], dtype=jnp.float32)) / jnp.sqrt(U.shape[1])
    vt_ortho = jnp.linalg.norm(Vt @ Vt.T.conj() - jnp.eye(Vt.shape[0], dtype=jnp.float32)) / jnp.sqrt(Vt.shape[0])
    jax.config.update("jax_default_matmul_precision", precision)
    return float(recon), float(u_ortho), float(vt_ortho)


def bench_algo(
    algo: AlgorithmSpec,
    gen_fn: Callable,
    key,
):
    key, subkey = jax.random.split(key)
    matrix = gen_fn(subkey)
    jax.config.update("jax_default_matmul_precision", algo.precision)
    for _ in range(5):
        u_w, u_s, u_v = algo.fn(matrix)
        u_w.block_until_ready()
        u_s.block_until_ready()
        u_v.block_until_ready()

    times_ms: List[float] = []
    recon_errors: List[float] = []
    u_errors: List[float] = []
    vt_errors: List[float] = []
    n_tries = 100 if gen_fn.shape[0] <= 4096 else 10
    for _ in range(n_tries):
        key, subkey = jax.random.split(key)
        matrix = gen_fn(subkey)
        start = time.perf_counter()
        u, s, vt = algo.fn(matrix)
        u.block_until_ready()
        s.block_until_ready()
        vt.block_until_ready()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        times_ms.append(elapsed_ms)
        recon_err, u_err, vt_err = compute_error(matrix, u, s, vt)
        recon_errors.append(recon_err)
        u_errors.append(u_err)
        vt_errors.append(vt_err)

    mean_time = mean(times_ms)
    spread = pstdev(times_ms) if len(times_ms) > 1 else 0.0
    recon_mean = mean(recon_errors)
    recon_std = pstdev(recon_errors) if len(recon_errors) > 1 else 0.0
    u_mean = mean(u_errors)
    u_std = pstdev(u_errors) if len(u_errors) > 1 else 0.0
    vt_mean = mean(vt_errors)
    vt_std = pstdev(vt_errors) if len(vt_errors) > 1 else 0.0

    return {
        "algorithm": algo.name,
        "n": gen_fn.shape[0],
        "m": gen_fn.shape[1],
        "cond": gen_fn.cond,
        "rank": gen_fn.rank,
        "mean_ms": float(mean_time),
        "std_ms": float(spread),
        "recon_error": float(recon_mean),
        "recon_error_std": float(recon_std),
        "u_ortho_error": float(u_mean),
        "u_ortho_error_std": float(u_std),
        "vt_ortho_error": float(vt_mean),
        "vt_ortho_error_std": float(vt_std),
    }

def generate_matrix(shape, cond, rank):
    assert shape[0] >= shape[1], "Rank should be square or tall"
    class Generator:
        def __init__(self):
            self.shape = shape
            self.cond = cond
            self.rank = rank
        def __call__(self, key):
            u = jax.lax.linalg.qr(jax.random.normal(key, (self.shape[0], self.shape[1]), dtype=jnp.float32), full_matrices=False)[0]
            v = jax.lax.linalg.qr(jax.random.normal(key, (self.shape[1], self.shape[1]), dtype=jnp.float32), full_matrices=False)[0]
            s = jnp.zeros(self.shape[1], dtype=jnp.float32)
            s = s.at[:self.rank].set(jnp.logspace(0, jnp.log10(self.cond), self.rank, dtype=jnp.float32))
            return jax.block_until_ready(u @ (jnp.diag(s) @ v))
    return Generator()

@jax.jit
def _svd_qr(A):
    return jax.lax.linalg.svd(A, algorithm=SvdAlgorithm.QR, full_matrices=False)
@jax.jit
def _svd_jacobi(A):
    return jax.lax.linalg.svd(A, algorithm=SvdAlgorithm.JACOBI, full_matrices=False)
@jax.jit
def _svd_polar(A):
    return jax.lax.linalg.svd(A, algorithm=SvdAlgorithm.POLAR, full_matrices=False)
@jax.jit
def _svd_qdwh(A):
    return svd(A, polar_method="qdwh")
@jax.jit
def _svd_cans_5(A):
    return svd(A, polar_method="cans", cans_tol=1e-5, eps_qr=1e-5)
@jax.jit
def _svd_cans_3(A):
    return svd(A, polar_method="cans", cans_tol=1e-3, eps_qr=1e-3)

algos = [
    AlgorithmSpec("CUDA QR", "float32", _svd_qr),
    AlgorithmSpec("CUDA JACOBI", "float32", _svd_jacobi),
    AlgorithmSpec("CUDA POLAR", "float32", _svd_polar),
    AlgorithmSpec("QDWH SVD", "float32", _svd_qdwh),
    AlgorithmSpec("CANS SVD (fp32)", "float32", _svd_cans_5),
    AlgorithmSpec("CANS SVD (tf32)", "tensorfloat32", _svd_cans_3),
    AlgorithmSpec("CANS SVD (bf16)", "bfloat16", _svd_cans_3),
]

matrices = [
    *[
        generate_matrix((size, size), 10, size) for size in
        [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    ],
    *[
        generate_matrix((size, size), 10, size // 10) for size in
        [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    ],
    *[
        generate_matrix((4096, size), 10, size) for size in
        [16, 32, 64, 128, 256, 512, 1024, 2048]
    ],
    *[
        generate_matrix((size, 4096), 10, 4096) for size in
        [8192, 16384, 32768, 65536]
    ],
]

def main():
    data = []
    key = jax.random.PRNGKey(0)
    for matrix in tqdm(matrices, desc="Matrices", unit="matrix"):
        for algo in tqdm(algos, desc="Algorithms", unit="algo", leave=False):
            result = bench_algo(algo, matrix, key)
            data.append(result)
    output_path = Path(os.getenv("OUTPUT_PATH", "bench_results.csv"))
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
