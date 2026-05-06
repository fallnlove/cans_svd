from functools import partial
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

from src_jax.baselines.eig_analytic import _eig_analytic_bwd
from src_jax.baselines.eig_original import _eig_original_bwd
from src_jax.baselines.eig_pi import _eig_pi_bwd
from src_jax.baselines.eig_taylor import _eig_taylor_bwd
from src_jax.baselines.eig_topn import _eig_topn_bwd
from src_jax.baselines.eig_pade import _eig_pade_bwd
from src_jax.baselines.eig_trunc import _eig_trunc_bwd
from src_jax.polar_decomposition import _polar_bwd, _solve_lyapunov_sym, polar_decomposition
from src_jax.svd import svd
from src_jax.baselines.svd_dl import svd_dl, _svd_dl_bwd
from src_jax.baselines.svd_townsend import svd_townsend, _svd_townsend_bwd
from src_jax.baselines.svd_inv import svd_inv, _svd_inv_bwd
from src_jax.baselines.svd_taylor1 import svd_taylor, _svd_taylor_bwd
from src_jax.svd_tikhonov import svd_tikhonov, _svd_tikhonov_bwd
from src_jax.svd_pseudo import svd_pseudo, _svd_pseudo_bwd


@dataclass
class AlgorithmSpec:
    name: str
    precision: str
    calc_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]


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

    times_ms: List[float] = []
    frob_norm = []
    spec_norm = []
    n_tries = 10
    for _ in range(n_tries):
        key, subkey = jax.random.split(key)
        matrix = gen_fn(subkey)
        start = time.perf_counter()
        out = algo.calc_fn(*matrix)
        out = jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        times_ms.append(elapsed_ms)
        frob_norm.append(jnp.linalg.norm(out[0], ord='fro'))
        spec_norm.append(jnp.linalg.norm(out[0], ord=2))

    mean_time = mean(times_ms)
    spread = pstdev(times_ms) if len(times_ms) > 1 else 0.0
    mean_frob = jnp.mean(jnp.array(frob_norm))
    std_frob = jnp.std(jnp.array(frob_norm)) if len(frob_norm) > 1 else 0.0
    mean_spec = jnp.mean(jnp.array(spec_norm))
    std_spec = jnp.std(jnp.array(spec_norm)) if len(spec_norm) > 1 else 0.0

    return {
        "algorithm": algo.name,
        "n": gen_fn.shape[0],
        "m": gen_fn.shape[1],
        "gap": gen_fn.gap,
        "rank": gen_fn.rank,
        "mean_ms": float(mean_time),
        "std_ms": float(spread),
        'frob_mean': float(mean_frob),
        'frob_std': float(std_frob),
        'spec_mean': float(mean_spec),
        'spec_std': float(std_spec),
    }

def generate_matrix(shape, gap, rank):
    assert shape[0] >= shape[1], "Rank should be square or tall"
    class Generator:
        def __init__(self):
            self.shape = shape
            self.gap = gap
            self.rank = rank
        def __call__(self, key):
            u = jax.lax.linalg.qr(jax.random.normal(key, (self.shape[0], self.shape[1]), dtype=jnp.float32), full_matrices=False)[0]
            v = jax.lax.linalg.qr(jax.random.normal(key, (self.shape[1], self.shape[1]), dtype=jnp.float32), full_matrices=False)[0]
            s = jnp.zeros(self.shape[1], dtype=jnp.float32)
            s = s.at[:self.rank].set(jnp.arange(1, self.rank + 1, dtype=jnp.float32))
            s = s.at[self.rank - 1].set(self.rank - 1 + self.gap)
            s = s[::-1]
            return 'cans', 1e-5, 1e-5, jax.block_until_ready((u, s, v.T.conj(), u @ jnp.diag(s) @ v.T.conj())), jax.block_until_ready((jnp.sign(u), jnp.sign(s), jnp.sign(v.T.conj())))
    return Generator()

algos = [
    AlgorithmSpec("Townsend Analytical", "float32", _svd_townsend_bwd),
    AlgorithmSpec("Taylor", "float32", _svd_taylor_bwd),
    AlgorithmSpec("Tikhonov (our)", "float32", _svd_tikhonov_bwd),
    AlgorithmSpec("Pseudo (our)", "float32", _svd_pseudo_bwd),
]

matrices = [
    *[
        generate_matrix((4096, 4096), gap, 4096) for gap in
        [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
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
