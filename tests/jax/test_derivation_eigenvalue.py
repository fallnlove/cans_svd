import jax
import jax.test_util
import jax.numpy as jnp
import pytest

from src_jax.eigenvalue_decomposition import symmetric_eigh
from src_jax.eigenvalue_decomposition_pseudo import symmetric_eigh_pseudo


jax.config.update("jax_enable_x64", True)

def random_matrix(m, n, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (m, n))

def make_spd(A, eps=0.1):
    return A @ A.T + eps * jnp.eye(A.shape[-1], device=A.device)


def test_spd_reconstruction():
    n = 5

    A_base = random_matrix(n, n)
    A = make_spd(A_base)

    eigvecs, eigvals = symmetric_eigh(A)

    A_rec = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

    assert jnp.allclose(A, A_rec, atol=1e-6)
    assert eigvals.shape == (n,)
    assert eigvecs.shape == (n, n)


def test_orthogonality():
    n = 6

    A = make_spd(random_matrix(n, n))
    eigvecs, eigvals = symmetric_eigh(A)

    I = jnp.eye(n)
    assert jnp.allclose(eigvecs.T @ eigvecs, I, atol=1e-6)



def test_backward_exists_and_symmetric():
    n = 5

    A_base = random_matrix(n, n)
    A = make_spd(A_base)

    def f(a):
        eigvecs, eigvals = symmetric_eigh(a)
        return jnp.sum(eigvecs)

    gradient = jax.grad(f)(A)

    assert gradient is not None
    assert jnp.allclose(gradient, gradient.T, atol=1e-8)
    assert jnp.linalg.norm(gradient) > 0


def test_gradient_identity_matrix():
    A = jnp.eye(3) * 2.0

    def f(A):
        eigvecs, eigvals = symmetric_eigh(A)
        return jnp.sum(eigvals)
    gradient = jax.grad(f)(A)

    assert gradient is not None
    assert jnp.allclose(gradient, jnp.eye(3), atol=1e-6)



def test_close_eigenvalues_series_stability():
    A = jnp.array(
        [[2.0, 0.0002, 0.0],
         [0.0, 2.0001, 0.0001],
         [0.000151, 0.00001, 5.0]],
    )
    eigvecs, eigvals = symmetric_eigh(A @ A.T)

    def f(A):
        eigvecs, eigvals = symmetric_eigh(A @ A.T)

        P = eigvecs[:, :2] @ eigvecs[:, :2].T
        return ((jnp.eye(3) - P) @ A).mean()
    
    gradient = jax.grad(f)(A)

    assert jnp.isfinite(eigvals).all()
    assert gradient is not None
    assert jnp.isfinite(gradient).all()



def test_against_torch_eigh_gradient_stability():
    A = jnp.array(
        [[2.0, 0.0002, 0.0],
         [0.0, 2.0001, 0.0001],
         [0.000151, 0.00001, 5.0]],
    )

    def f(A):
        eigvecs, eigvals = symmetric_eigh(A @ A.T)

        return ((jnp.eye(3) - eigvecs[:, 1:] @ eigvecs[:, 1:].T) @ A).mean()
    gradient = jax.grad(f)(A)

    assert gradient is not None
    assert jnp.isfinite(gradient).all()


def test_gradcheck_eigh():
    n = 3
    B = random_matrix(n, n)
    A = (B @ B.T + 1 * jnp.eye(n))

    def f(X):
        eigvals, eigvecs = symmetric_eigh(X)
        return eigvals.sum() + eigvecs.sum()

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=['rev'],
        eps=1e-8,
        atol=1e-4,
        rtol=1e-4,
    )

def test_gradcheck_eigh_pseudo():
    n = 3
    B = random_matrix(n, n)
    A = (B @ B.T + 1 * jnp.eye(n))

    def f(X):
        eigvals, eigvecs = symmetric_eigh_pseudo(X)
        return eigvals.sum() + eigvecs.sum()

    jax.test_util.check_grads(
        f,
        (A,),
        order=1,
        modes=['rev'],
        eps=1e-8,
        atol=1e-4,
        rtol=1e-4,
    )
