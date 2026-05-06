import jax
import jax.numpy as jnp
from functools import partial

import numpy as np

jnp.set_printoptions(precision=20)
jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["dtype"])
def explicit3(A, B, dtype=jnp.float64):
    # explicit formula for optimal 3-rd order polynomial on the segment [A, B]
    e = jnp.sqrt((A**2 + A * B + B**2) / 3)
    a = 2 / (2 * e**3 + A**2 * B + B**2 * A)
    p = jnp.array([-a, 0, a * (A**2 + A * B + B**2), 0])[::-1]
    err = (2 * e**3 - A**2 * B - B**2 * A) / (2 * e**3 + A**2 * B + B**2 * A)
    return p.astype(dtype), err.astype(dtype)


@partial(jax.jit, static_argnames=["n", "degree", "dtype"])
def delta_orthogonalization(n=1, degree=3, delta=0.3, B=1., dtype=jnp.float32):
    # find composition of n polynomials of specified degree on the interval
    # [0, B], which falls into [1-delta, 1+delta]
    # the derivative of composition at zero is maximized
    if degree != 3:
        raise NotImplementedError
    Al = 0.0
    Ar = B
    e = 100.0

    def cond_fun(val):
        e, _, _, _ = val
        return jnp.abs(e - delta) > 1e-7

    def body_fun(val):
        e, Al, Ar, lst = val
        a, b = (Al + Ar) / 2, B
        lst = []
        for _ in range(n):
            Q, e = explicit3(a, b, jnp.float64)
            lst.append(Q)
            a, b = 1 - e, 1 + e
        Ar = jnp.where(e < delta, (Ar + Al) / 2, Ar)
        Al = jnp.where(e < delta, Al, (Al + Ar) / 2)
        return e, Al, Ar, lst

    e, Al, Ar, lst = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (e,
         Al,
         Ar,
         [jnp.zeros(degree + 1, dtype=jnp.float64) for _ in range(n)]),
    )
    return jnp.stack([i.astype(dtype) for i in lst]), \
        ((Al + Ar) / 2).astype(dtype)


@partial(
    jax.jit,
    static_argnames=[
        "max_iter",
        "degree",
        "preprocess",
        "preprocess_iters",
        "denom_type",
        "tol",
    ],
)
def cans_iteration(
    A,
    max_iter=50,
    a=1e-8,
    degree=3,
    preprocess=True,
    preprocess_iters=2,
    delta=0.99,
    denom_type="one_inf",
    tol=None,
):
    # CANS iteration for orthogonalization
    # a is the left boundary of the segment
    # max_iter is the maximum number of iterations
    if degree != 3:
        raise NotImplementedError
    if tol is None:
        tol = 1e-5
        if jax.config.jax_default_matmul_precision == "tensorfloat32":
            tol = 1e-3
    n_start = A.shape[0]
    if A.shape[0] < A.shape[1]:
        A = A.T
    else:
        A = A.copy()
    b = 1  # assume that matrix is normalized
    err = 100000
    n = A.shape[1]
    id = jnp.eye(n, dtype=A.dtype)
    A2 = A.T @ A
    # A3 = A @ A2
    if denom_type == "one_inf":
        one_norm = jnp.linalg.norm(A, ord=1)
        inf_norm = jnp.linalg.norm(A, ord=np.inf)
        alpha_inverse = jax.lax.rsqrt(one_norm) * jax.lax.rsqrt(inf_norm)
        alpha_inverse = jnp.where(one_norm == 0, 1, alpha_inverse)
    # elif denom_type == "frob":
        # norm = jnp.linalg.norm(A3) ** (1/3)
        # alpha_inverse = jnp.where(norm == 0, 1, 1 / norm)
    else:
        raise NotImplementedError

    A2 *= alpha_inverse ** 2
    # A3 *= alpha_inverse ** 3
    A = A * alpha_inverse

    if preprocess:
        lst, _ = delta_orthogonalization(
            preprocess_iters,
            degree,
            delta,
            dtype=A.dtype
        )

        def body_prep(i, val):
            A, A2 = val
            A3 = A @ A2
            A = lst[i][1] * A + lst[i][3] * A3
            A2 = A.T @ A
            return A, A2

        A, A2 = jax.lax.fori_loop(
            0, preprocess_iters, body_prep, (A, A2)
        )
        a, b = 1 - delta, 1 + delta

    cnt = 0
    err = jnp.linalg.norm(A2 - id) / jnp.sqrt(n)

    def cond_fun(val):
        return jnp.logical_and(val[0] < max_iter, val[1] > tol)

    def body_fun(val):
        cnt, err, A, a, b, A2 = val
        A3 = A @ A2
        p, e = explicit3(a, b, dtype=A.dtype)
        a, b = 1 - e, 1 + e
        A = p[1] * A + p[3] * A3
        A2 = A.T @ A
        err = jnp.linalg.norm(A2 - id) / jnp.sqrt(n)
        return cnt + 1, err, A, a, b, A2

    cnt, err, A, a, b, A2 = jax.lax.while_loop(
        cond_fun, body_fun, (cnt, err, A, a, b, A2)
    )

    return A if n_start == A.shape[0] else A.T
