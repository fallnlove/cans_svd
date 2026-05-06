import jax
import jax.test_util
import jax.numpy as jnp
import pytest

from src_jax.svd import svd
from src_jax.baselines.svd_townsend import svd_townsend
from src_jax.baselines.svd_taylor import svd_taylor
from src_jax.baselines.svd_inv import svd_inv
from src_jax.baselines.svd_dl import svd_dl
from src_jax.svd_tikhonov import svd_tikhonov
from src_jax.svd_pseudo import svd_pseudo

jax.config.update("jax_enable_x64", True)


def random_matrix(m, n, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (m, n))



def test_shapes_tall_matrix():
    A = random_matrix(20, 10)

    U, S, Vt = svd(A)

    assert U.shape == (20, 10)
    assert S.shape == (10,)
    assert Vt.shape == (10, 10)


def test_shapes_wide_matrix():
    A = random_matrix(10, 20)

    U, S, Vt = svd(A)

    assert U.shape == (10, 10)
    assert S.shape == (10,)
    assert Vt.shape == (10, 20)



def test_orthogonality():
    A = random_matrix(30, 15)

    U, S, Vt = svd(A)

    Iu = U.T @ U
    Iv = Vt @ Vt.T

    assert jnp.allclose(Iu, jnp.eye(15), atol=1e-6)
    assert jnp.allclose(Iv, jnp.eye(15), atol=1e-6)



def test_reconstruction():
    A = random_matrix(25, 12)

    U, S, Vt = svd(A)

    A_hat = U @ jnp.diag(S) @ Vt
    rel_err = jnp.linalg.norm(A - A_hat) / jnp.linalg.norm(A)

    assert rel_err < 1e-6



def test_singular_values_sorted():
    A = random_matrix(40, 10)

    _, S, _ = svd(A)

    assert jnp.all(S[:-1] >= S[1:])



def test_low_rank_matrix():

    m, n, r = 30, 20, 5
    U0, _ = jnp.linalg.qr(random_matrix(m, r, 0))
    V0, _ = jnp.linalg.qr(random_matrix(n, r, 42))
    S0 = jnp.linspace(5.0, 1.0, r)

    A = U0 @ jnp.diag(S0) @ V0.T

    U, S, Vt = svd(A)

    assert jnp.allclose(S[r:], jnp.zeros_like(S[r:]), atol=1e-6)

    A_hat = U @ jnp.diag(S) @ Vt
    assert jnp.allclose(A, A_hat, atol=1e-6)



def test_backward_runs():
    A = random_matrix(15, 10)

    gradient = jax.grad(lambda x: jnp.sum(jnp.abs(svd(x)[0])))(A)

    assert gradient is not None
    assert jnp.isfinite(gradient).all()


@pytest.mark.parametrize(
    "shape",
    [
        (6, 4),
        (4, 6),
    ],
)
def test_gradcheck_svd(shape):

    A = random_matrix(*shape).T

    def f(A):
        U, S, Vt = svd(A)
        return jnp.sum(jnp.abs(U)) + jnp.sum(jnp.abs(S)) + jnp.sum(jnp.abs(Vt))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (6, 4),
        (4, 6),
    ],
)
def test_gradcheck_svd_taylor(shape):

    A = random_matrix(*shape).T

    def f(A):
        U, S, Vt = svd_taylor(A)
        return jnp.sum(jnp.abs(U)) + jnp.sum(jnp.abs(S)) + jnp.sum(jnp.abs(Vt))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (6, 4),
        (4, 6),
    ],
)
def test_gradcheck_svd_townsend(shape):

    A = random_matrix(*shape).T

    def f(A):
        U, S, Vt = svd_townsend(A)
        return jnp.sum(jnp.abs(U)) + jnp.sum(jnp.abs(S)) + jnp.sum(jnp.abs(Vt))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (6, 4),
        (4, 6),
    ],
)
def test_gradcheck_svd_inv(shape):

    A = random_matrix(*shape).T

    def f(A):
        U, S, Vt = svd_inv(A)
        return jnp.sum(jnp.abs(U)) + jnp.sum(jnp.abs(S)) + jnp.sum(jnp.abs(Vt))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (4, 6),
    ],
)
def test_gradcheck_svd_dl(shape):

    A = random_matrix(*shape).T

    def f(A):
        #  This method can not handle gradients on U
        U, S, Vt = svd_dl(A)
        return jnp.sum(jnp.abs(S)) + jnp.sum(jnp.abs(Vt))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (6, 4),
    ],
)
def test_gradcheck_svd_tikhonov(shape):

    A = random_matrix(*shape).T

    def f(A):
        U, S, Vt = svd_tikhonov(A)
        return jnp.sum(jnp.abs(U)) + jnp.sum(jnp.abs(S)) + jnp.sum(jnp.abs(Vt))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (6, 4),
    ],
)
def test_gradcheck_svd_pseudo(shape):

    A = random_matrix(*shape).T

    def f(A):
        U, S, Vt = svd_pseudo(A)
        return jnp.sum(jnp.abs(U)) + jnp.sum(jnp.abs(S)) + jnp.sum(jnp.abs(Vt))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )

