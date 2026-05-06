from contextlib import contextmanager
from typing import Literal, Optional


Backend = Literal["jax", "torch"]
Differentiation = Literal["tikhonov", "pseudo"]
MatmulPrecision = Literal["fp32", "tf32"]

_EPS_BY_PRECISION = {
    "fp32": 1e-5,
    "tf32": 1e-3,
}


def _normalize_backend(backend: str) -> str:
    aliases = {
        "torch": "torch",
        "pytorch": "torch",
        "jax": "jax",
    }
    try:
        return aliases[backend.lower()]
    except KeyError as exc:
        raise ValueError("backend must be 'torch' or 'jax'") from exc


def _normalize_differentiation(differentiation: str) -> str:
    aliases = {
        "tikhonov": "tikhonov",
        "tik": "tikhonov",
        "pseudo": "pseudo",
        "pinv": "pseudo",
    }
    try:
        return aliases[differentiation.lower()]
    except KeyError as exc:
        raise ValueError(
            "differentiation must be 'tikhonov', or 'pseudo'"
        ) from exc


def _normalize_matmul_precision(matmul_precision: str) -> str:
    precision = matmul_precision.lower()
    if precision not in _EPS_BY_PRECISION:
        raise ValueError("matmul_precision must be 'fp32' or 'tf32'")
    return precision


@contextmanager
def _torch_matmul_precision(precision: str):
    import torch

    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32

    try:
        torch.set_float32_matmul_precision("highest" if precision == "fp32" else "high")
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


@contextmanager
def _jax_matmul_precision(precision: str):
    import jax

    try:
        old_precision = jax.config.read("jax_default_matmul_precision")
    except Exception:
        old_precision = None

    jax_precision = "float32" if precision == "fp32" else "tensorfloat32"
    try:
        jax.config.update("jax_default_matmul_precision", jax_precision)
        yield
    finally:
        if old_precision is not None:
            jax.config.update("jax_default_matmul_precision", old_precision)


def _torch_svd(
    matrix,
    differentiation: str,
    cans_tol: float,
    eps_qr: float,
):
    if differentiation == "tikhonov":
        from src_torch.derivation.svd_tikhonov import svd_tikhonov

        return svd_tikhonov(
            matrix,
            cans_tol=cans_tol,
            eps_qr=eps_qr,
        )

    from src_torch.derivation.svd_pseudo import svd_pseudo

    return svd_pseudo(
        matrix,
        cans_tol=cans_tol,
        eps_qr=eps_qr,
    )


def _jax_svd(
    matrix,
    differentiation: str,
    cans_tol: float,
    eps_qr: float,
):
    if differentiation == "tikhonov":
        from src_jax.svd_tikhonov import svd_tikhonov as svd_impl
    else:
        from src_jax.svd_pseudo import svd_pseudo as svd_impl

    return svd_impl(
        matrix,
        cans_tol=cans_tol,
        eps_qr=eps_qr,
    )


def svd(
    matrix,
    *,
    backend: Backend = "jax",
    differentiation: Differentiation = "tikhonov",
    matmul_precision: MatmulPrecision = "fp32",
):
    """
    Compute CANS-based SVD with a selected backend and backward rule.

    Parameters
    ----------
    matrix
        Torch or JAX matrix.
    backend
        "torch" or "jax".
    differentiation
        "tikhonov", or "pseudo".
    matmul_precision
        "fp32" uses regular fp32 matmul precision and eps_qr=1e-5 by default.
        "tf32" enables TF32-style matmul precision where the backend supports it
        and eps_qr=1e-3 by default.
    """
    backend = _normalize_backend(backend)
    differentiation = _normalize_differentiation(differentiation)
    matmul_precision = _normalize_matmul_precision(matmul_precision)

    if backend == "torch":
        with _torch_matmul_precision(matmul_precision):
            return _torch_svd(
                matrix,
                differentiation=differentiation,
                cans_tol=_EPS_BY_PRECISION[matmul_precision],
                eps_qr=_EPS_BY_PRECISION[matmul_precision],
            )

    with _jax_matmul_precision(matmul_precision):
        return _jax_svd(
            matrix,
            differentiation=differentiation,
            cans_tol=_EPS_BY_PRECISION[matmul_precision],
            eps_qr=_EPS_BY_PRECISION[matmul_precision],
        )
