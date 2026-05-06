# Fast Differentiable SVD on GPU via Polar Decomposition

Fast, differentiable Singular Value Decomposition (SVD) implementations based on CANS iteration for polar decomposition. The library exposes one small public API that can dispatch to JAX or PyTorch and choose the backward rule.

## Install And Use

Download code from repository and run following commands:

```bash
cd cans_svd
pip install .
```

For local development:

```bash
pip install -e .
```

The package installs the runtime dependencies `jax`, `numpy`, and `torch`.

See [demo.ipynb](demo.ipynb) for a notebook demo.

### PyTorch

```python
import torch
from cans_svd import svd

A = torch.randn(128, 64, device="cuda", requires_grad=True)

U, S, Vt = svd(
    A,
    backend="torch",
    differentiation="tikhonov",   # "tikhonov" or "pseudo"
    matmul_precision="fp32",      # "fp32" or "tf32"
)

loss = S.sum()
loss.backward()
```

### JAX

```python
import jax
import jax.numpy as jnp
from cans_svd import svd

key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (128, 64))

U, S, Vt = svd(
    A,
    backend="jax",
    differentiation="pseudo",     # "tikhonov" or "pseudo"
    matmul_precision="tf32",      # "fp32" or "tf32"
)
```

`matmul_precision="fp32"` uses `cans_tol=eps_qr=1e-5`; `matmul_precision="tf32"` uses `cans_tol=eps_qr=1e-3`. The internal backward `eps` and singular-value `threshold` in the regularized SVD implementations are fixed at `1e-8`.

## Reproducing Experiments

This section is for reproducing the benchmark experiments from the repository.

```bash
cd cans_svd
pip install -r requirements.txt
```

Then run:

```bash
python bench.py
python bench_backward.py
python bench_lyapunov.py
```

## Project Structure

- `src/cans_svd/` - pip-installable public library API.
- `src_jax/` - JAX implementations and baselines.
- `src_torch/` - PyTorch implementations and baselines.
- `tests/` - JAX and PyTorch test suites.
