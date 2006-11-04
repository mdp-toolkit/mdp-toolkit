from routines import timediff, refcast, scast, rotate, random_rot, \
     symrand, norm2, uniq, ordered_uniq, cov2, \
     comb, sqrtm, SymeigException
from introspection import dig_node, get_node_size
from quad_forms import QuadraticForm
from covariance import CovarianceMatrix, DelayCovarianceMatrix
from progress_bar import progressinfo
import mdp as _mdp


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
solve = lambda x,y: refcast(_solve(x,y), x.dtype)

def svd(x, _mdp=_mdp):
    tc = x.dtype
    u,s,v = _mdp.numx_linalg.svd(x)
    return refcast(u, tc), refcast(s, tc), refcast(v, tc)

# clean up namespace
del routines
del introspection
del quad_forms
del covariance
del progress_bar
