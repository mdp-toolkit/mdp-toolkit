from routines import (timediff, refcast, scast, rotate, random_rot,
                      permute, symrand, norm2, uniq, ordered_uniq, cov2,
                      mult_diag, comb, sqrtm, get_dtypes, nongeneral_svd,
                      SymeigException, hermitian, _symeig_fake)
from introspection import dig_node, get_node_size
from quad_forms import QuadraticForm
from covariance import (CovarianceMatrix, DelayCovarianceMatrix,
                        MultipleCovarianceMatrices)
from progress_bar import progressinfo
import mdp as _mdp

try:
    import symeig
    SymeigException = symeig.SymeigException
    symeig = symeig.symeig
except ImportError:
    symeig = routines._symeig_fake

# matrix multiplication function
# we use an alias to be able to use the wrapper for the 'gemm' Lapack
# function in the future
mult = _mdp.numx.dot

# workaround to numpy issues with dtype behavior:
# 'f' is upcasted at least in the following functions
_inv = _mdp.numx_linalg.inv
inv = lambda x: refcast(_inv(x), x.dtype)
_pinv = _mdp.numx_linalg.pinv
pinv = lambda x: refcast(_pinv(x), x.dtype)
_solve = _mdp.numx_linalg.solve
solve = lambda x, y: refcast(_solve(x, y), x.dtype)

def svd(x):
    """Wrap the numx SVD routine, so that it returns arrays of the correct
    dtype and a SymeigException in case of failures."""
    tc = x.dtype
    try:
        u, s, v = _mdp.numx_linalg.svd(x)
    except _mdp.numx_linalg.LinAlgError, exc:
        raise SymeigException(str(exc))
    return refcast(u, tc), refcast(s, tc), refcast(v, tc)

# clean up namespace
del routines
del introspection
del quad_forms
del covariance
del progress_bar

__all__ = ['CovarianceMatrix', 'DelayCovarianceMatrix',
           'MultipleCovarianceMatrices', 'QuadraticForm', 'SymeigException',
           'comb', 'cov2', 'dig_node', 'get_dtypes', 'get_node_size',
           'hermitian', 'inv', 'mult', 'mult_diag', 'nongeneral_svd',
           'norm2', 'ordered_uniq', 'permute', 'pinv', 'progressinfo',
           'random_rot', 'refcast', 'rotate', 'scast', 'solve', 'sqrtm',
           'svd', 'symeig', 'symrand', 'timediff', 'uniq']
