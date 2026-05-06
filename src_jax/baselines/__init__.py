'''
Baseline differentiable eigendecomposition implementations in JAX.

This module contains JAX implementations of various differentiable eigendecomposition algorithms,
extracted from CV model pipelines. Each implementation provides forward and backward passes
for the eigendecomposition operation with different gradient computation strategies.

All implementations work on symmetric square matrices (input is assumed to be symmetric)
and return eigenvectors U and eigenvalues S.
The backward pass uses gradients w.r.t. both eigenvectors and eigenvalues.
'''

from .eig_pade import eig_pade
from .eig_taylor import eig_taylor
from .eig_pi import eig_pi
from .eig_original import eig_original
from .eig_trunc import eig_trunc
from .eig_topn import eig_topn
from .eig_analytic import eig_analytic

__all__ = [
    'eig_pade',
    'eig_taylor',
    'eig_pi',
    'eig_original',
    'eig_trunc',
    'eig_topn',
    'eig_analytic',
]
