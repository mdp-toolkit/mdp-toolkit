"""These are some benchmark functions for MDP.

Run them with:

>>> import mdp
>>> mdp.test.benchmark()

"""

import mdp
from mdp.utils import symeig, mult

numx = mdp.numx
numx_rand = mdp.numx_rand

def _random_spd_mtx(dim):
    rnd = numx_rand.random((dim,dim))
    v = mdp.utils.random_rot(rnd)
    d = mdp.numx.diag(numx_rand.random(dim) + 1)
    return  mult(v,mult(d,v.T))

####### benchmark function

def matmult_c_MDP_benchmark(dim):
    """    This benchmark multiplies two contiguous matrices using the
    MDP internal matrix multiplication routine.
    First argument matrix dimensionality"""
    a = numx_rand.random((dim,dim))
    b = numx_rand.random((dim,dim))
    out = mdp.utils.mult(a,b)

def matmult_c_scipy_benchmark(dim):
    """    This benchmark multiplies two contiguous matrices using the
    scipy internal matrix multiplication routine.
    First argument matrix dimensionality"""
    a = numx_rand.random((dim,dim))
    b = numx_rand.random((dim,dim))
    out = numx.dot(a,b)
    
def matmult_n_MDP_benchmark(dim):
    """    This benchmark multiplies two non-contiguous matrices using the
    MDP internal matrix multiplication routine.
    First argument matrix dimensionality"""
    a = numx_rand.random((dim,dim)).T
    b = numx_rand.random((dim,dim)).T
    out = mdp.utils.mult(a,b)

def matmult_n_scipy_benchmark(dim):
    """    This benchmark multiplies two non-contiguous matrices using the
    scipy internal matrix multiplication routine.
    First argument matrix dimensionality"""
    a = numx_rand.random((dim,dim)).T
    b = numx_rand.random((dim,dim)).T
    out = numx.dot(a,b)

def matmult_cn_MDP_benchmark(dim):
    """    This benchmark multiplies a contiguous matrix with a
    non-contiguous matrix using the MDP internal matrix multiplication
    routine.
    First argument matrix dimensionality"""
    a = numx_rand.random((dim,dim)).T
    b = numx_rand.random((dim,dim))
    out = mdp.utils.mult(a,b)

def matmult_cn_scipy_benchmark(dim):
    """    This benchmark multiplies a contiguous matrix with a
    non-contiguous matrix using the scipy internal matrix multiplication
    routine.
    First argument matrix dimensionality"""
    a = numx_rand.random((dim,dim)).T
    b = numx_rand.random((dim,dim))
    out = numx.dot(a,b)

def quadratic_expansion_benchmark(dim, len, times):
    """    This benchmark expands random data of shape (len, dim)
    'times' times.
    Arguments: (dim,len,times)."""
    a = numx_rand.random((len,dim))
    qnode = mdp.nodes.QuadraticExpansionNode()
    for i in xrange(times):
        qnode(a)
        
def polynomial_expansion_benchmark(dim, len, degree, times):
    """    This benchmark expands random data of shape (len, dim)
    'times' times in the space of polynomials of degree 'degree'.
    Arguments: (dim,len,degree,times)."""
    a = numx_rand.random((len,dim))
    pnode = mdp.nodes.PolynomialExpansionNode(degree)
    for i in xrange(times):
        pnode(a)

# ISFA benchmark

def _tobias_mix(src):
    mix = src.copy()
    mix[:,0]=(src[:,1]+3*src[:,0]+6)*numx.cos(1.5*numx.pi*src[:,0])
    mix[:,1]=(src[:,1]+3*src[:,0]+6)*numx.sin(1.5*numx.pi*src[:,0])
    return mix

def isfa_spiral_benchmark():
    # create independent sources
    nsrc = 2
    src = numx_rand.laplace(size=(50000,nsrc))
    fsrc = numx.fft.rfft(src,axis=0)
    # enforce different time scales
    for i in range(nsrc):
        fsrc[5000+(i+1)*1000:,i] = 0.
    src = numx.fft.irfft(fsrc,axis=0)
    # subtract mean and rescale between -1 and 1
    src -= src.mean(axis=0)
    src /= abs(src).max()
    # apply nonlinear "twist" transformation
    exp_src = _tobias_mix(src)
    # train
    flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(5),
                     mdp.nodes.ISFANode(lags=30, whitened=False,
                                        sfa_ica_coeff=[1.,100.],
                                        output_dim=2, verbose=True)])
    flow.train([None, exp_src])
    # get output signal
    out = flow(exp_src)
    return src, out

####### /benchmark function

POLY_EXP_ARGS = [(2**i, 100, j, 200) for j in range(2,5) for i in range(2,4)]

if mdp.numx_description in ['symeig', 'scipy', 'numpy']:
    MUL_MTX_DIMS = [[2**i] for i in range(4,11)]
    # list of (benchmark function, list of arguments)
    BENCH_FUNCS = [(matmult_c_MDP_benchmark, MUL_MTX_DIMS),
                   (matmult_c_scipy_benchmark, MUL_MTX_DIMS),
                   (matmult_n_MDP_benchmark, MUL_MTX_DIMS),
                   (matmult_n_scipy_benchmark, MUL_MTX_DIMS),
                   (matmult_cn_MDP_benchmark, MUL_MTX_DIMS),
                   (matmult_cn_scipy_benchmark, MUL_MTX_DIMS),
                   (polynomial_expansion_benchmark, POLY_EXP_ARGS)]
else:
    BENCH_FUNCS = [(polynomial_expansion_benchmark, POLY_EXP_ARGS)]

def get_benchmarks():
    return BENCH_FUNCS

if __name__ == "__main__":
    from testing_tools import run_benchmarks
    #num_rand.seed(1268049219)
    print "Running benchmarks: "
    run_benchmarks(get_benchmarks())
    print "Random seed was: ",numx_rand.get_state()[1][0]
