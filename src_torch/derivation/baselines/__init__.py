'''
Baseline differentiable eigendecomposition implementations from the paper "On the Eigenvalues of Global Covariance Pooling for Fine-grained Visual Recognition".

This module contains standalone implementations of various differentiable eigendecomposition algorithms,
extracted from CV model pipelines. Each implementation provides forward and backward passes
for the eigendecomposition operation with different gradient computation strategies.

All implementations work on symmetric square matrices (input is assumed to be symmetric)
and return eigenvectors U and eigenvalues S.
The backward pass only requires gradients w.r.t. eigenvalues (grad_S), not eigenvectors.
'''

from .eig_pade import EIG_Pade
from .eig_taylor import EIG_Taylor
from .eig_pi import EIG_PI
from .eig_original import EIG_Original
from .eig_trunc import EIG_Trunc
from .eig_topn import EIG_TopN

__all__ = [
    'EIG_Pade',
    'EIG_Taylor',
    'EIG_PI',
    'EIG_Original',
    'EIG_Trunc',
    'EIG_TopN',
]
