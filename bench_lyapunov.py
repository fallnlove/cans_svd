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
        out = algo.fn(*matrix)
        jax.block_until_ready(out)

    times_ms: List[float] = []
    n_tries = 100 if gen_fn.shape[0] <= 4096 else 10
    for _ in range(n_tries):
        key, subkey = jax.random.split(key)
        matrix = gen_fn(subkey)
        start = time.perf_counter()
        out = algo.fn(*matrix)
        jax.block_until_ready(out)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        times_ms.append(elapsed_ms)

    mean_time = mean(times_ms)
    spread = pstdev(times_ms) if len(times_ms) > 1 else 0.0

    return {
        "algorithm": algo.name,
        "n": gen_fn.shape[0],
        "m": gen_fn.shape[1],
        "cond": gen_fn.cond,
        "rank": gen_fn.rank,
        "mean_ms": float(mean_time),
        "std_ms": float(spread),
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
            return jax.block_until_ready(u @ (jnp.diag(s) @ u.T.conj())), jax.block_until_ready(v @ (jnp.diag(s) @ v.T.conj()))
    return Generator()

@jax.jit
def _solve_lyapunov_sym(A, H):
    w, Q = jnp.linalg.eigh(H)
    w += 1e-8  # for numerical stability

    Qt = Q.T
    A_tilde = Q.T @ A @ Q

    denom = w[:, None] + w[None, :]
    X_tilde = A_tilde / denom

    X = Q @ X_tilde @ Qt

    return X
@jax.jit
def _solve_lyapunov_sym_iter(A, H):
    C = A / jnp.linalg.norm(H, ord='fro')
    B = H / jnp.linalg.norm(H, ord='fro')

    def body_fun(carry):
        B, C, err, it = carry
        B2 = B @ B
        CB = C @ B
        B_new = 0.5 * B @ (3 * jnp.eye(B.shape[-1], dtype=B.dtype) - B2)
        C_new = 0.5 * (-B2 @ C + B @ CB + (3 * C - CB @ B))
        err_new = jnp.linalg.norm(B_new - jnp.eye(B.shape[-1], dtype=B.dtype), ord='fro') / jnp.linalg.norm(jnp.eye(B.shape[-1], dtype=B.dtype), ord='fro')
        return B_new, C_new, err_new, it + 1
    def cond_fun(carry):
        _, _, err, it = carry
        return jnp.logical_and(err > 1e-6, it < 100)
    
    _, C, _, _ = jax.lax.while_loop(cond_fun, body_fun, (B, C, jnp.inf, 0))

    X = C / 2

    
    return X

algos = [
    AlgorithmSpec("EIGH Based Approach", "float32", _solve_lyapunov_sym),
    AlgorithmSpec("Iterative Method", "float32", _solve_lyapunov_sym_iter),
]

matrices = [
    *[
        generate_matrix((4096, 4096), cond, 4096) for cond in
        [1.1, 1.5, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
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
