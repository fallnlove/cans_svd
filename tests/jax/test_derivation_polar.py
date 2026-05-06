import jax
import jax.test_util
import jax.numpy as jnp
import pytest

from src_jax.svd import svd
from src_jax.polar_decomposition import polar_decomposition
from src_jax.polar_decomposition_pseudo import polar_decomposition_pseudo
jax.config.update("jax_enable_x64", True)


def random_matrix(m, n, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (m, n))


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (6, 4),
    ],
)
def test_gradcheck_polar_decomposition(shape):

    A = random_matrix(*shape)

    def f(A):
        W, H = polar_decomposition(A)
        return jnp.sum(jnp.abs(W)) + jnp.sum(jnp.abs(H))

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
def test_gradcheck_polar_decomposition_pseudo(shape):

    A = random_matrix(*shape)

    def f(A):
        W, H = polar_decomposition_pseudo(A)
        return jnp.sum(jnp.abs(W)) + jnp.sum(jnp.abs(H))

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=["rev"],
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )

