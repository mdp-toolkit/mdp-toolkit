import mdp

# import numeric module (scipy, Numeric or numarray)
numx, numx_rand, numx_linalg  = mdp.numx, mdp.numx_rand, mdp.numx_linalg
import random

class SymeigException(mdp.MDPException):
    pass

def timediff(data):
    """Returns the array of the time differences of data."""
    # this is the fastest way we found so far
    return data[1:]-data[:-1]

def refcast(array, dtype):
    """
    Cast the array to dtype only if necessary, otherwise return a reference.
    """
    dtype = mdp.numx.dtype(dtype)
    if array.dtype == dtype:
        return array
    return array.astype(dtype)

def scast(scalar, dtype):
    """Convert a scalar in a 0D array of the given dtype."""
    return numx.array(scalar, dtype=dtype)

def rotate(mat, angle, columns = (0, 1), units = 'radians'):
    """
    Rotate in-place data matrix (NxM) in the plane defined by the columns=[i,j]
    when observation are stored on rows. Observations are rotated
    counterclockwise. This corresponds to the following matrix-multiplication
    for each data-point (unchanged elements omitted):

     [  cos(angle) -sin(angle)     [ x_i ]
        sin(angle)  cos(angle) ] * [ x_j ] 

    If M=2, columns=[0,1].
    """
    if units is 'degrees':
        angle = angle/180.*numx.pi
    cos_ = numx.cos(angle)
    sin_ = numx.sin(angle)
    [i, j] = columns
    col_i = mat[:, i] + 0.
    col_j = mat[:, j]
    mat[:, i] = cos_*col_i - sin_*col_j
    mat[:, j] = sin_*col_i + cos_*col_j

def permute(x, indices=(0, 0), rows=0, cols=1):
    """Swap two columns and (or) two rows of 'x', whose indices are specified
    in indices=[i,j].
    Note: permutations are done in-place. You'll lose your original matrix"""
    ## the nicer option:
    ## x[i,:],x[j,:] = x[j,:],x[i,:]
    ## does not work because array-slices are references.
    ## The following would work:
    ## x[i,:],x[j,:] = x[j,:].tolist(),x[i,:].tolist()
    ## because list-slices are copies, but you get 2
    ## copies instead of the one you need with our method.
    ## This would also work:
    ## tmp = x[i,:].copy()
    ## x[i,:],x[j,:] = x[j,:],tmp
    ## but it is slower (for larger matrices) than the one we use.
    [i, j] = indices
    if rows:
        x[i, :], x[j, :] = x[j, :], x[i, :] + 0
    if cols:
        x[:, i], x[:, j] = x[:, j], x[:, i] + 0

def hermitian(x):
    """Compute the Hermitian, i.e. conjugate transpose, of x."""
    return x.T.conj()

def symrand(dim_or_eigv, dtype="d"):
    """Return a random symmetric (Hermitian) matrix.
    
    If 'dim_or_eigv' is an integer N, return a NxN matrix, with eigenvalues
        uniformly distributed on (-1,1).
        
    If 'dim_or_eigv' is  1-D real array 'a', return a matrix whose
                      eigenvalues are 'a'.
    """
    if isinstance(dim_or_eigv, int):
        dim = dim_or_eigv
        d = (numx_rand.random(dim)*2) - 1
    elif isinstance(dim_or_eigv,
                    numx.ndarray) and len(dim_or_eigv.shape) == 1:
        dim = dim_or_eigv.shape[0]
        d = dim_or_eigv
    else:
        raise mdp.MDPException("input type not supported.")
    
    v = random_rot(dim)
    #h = mdp.utils.mult(mdp.utils.mult(hermitian(v), mdp.numx.diag(d)), v)
    h = mdp.utils.mult(mult_diag(d, hermitian(v), left=False), v)
    # to avoid roundoff errors, symmetrize the matrix (again)
    h = 0.5*(h.T+h)
    if dtype in ('D', 'F', 'G'):
        h2 = symrand(dim_or_eigv)
        h = h + 1j*(numx.triu(h2)-numx.tril(h2))
    return refcast(h, dtype)

def random_rot(dim, dtype='d'):
    """Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization"""
    H = mdp.numx.eye(dim, dtype=dtype)
    D = mdp.numx.ones((dim,), dtype=dtype)
    for n in range(1, dim):
        x = mdp.numx_rand.normal(size=(dim-n+1,)).astype(dtype)
        D[n-1] = mdp.numx.sign(x[0])
        x[0] -= D[n-1]*mdp.numx.sqrt((x*x).sum())
        # Householder transformation
        Hx = ( mdp.numx.eye(dim-n+1, dtype=dtype)
               - 2.*mdp.numx.outer(x, x)/(x*x).sum() )
        mat = mdp.numx.eye(dim, dtype=dtype)
        mat[n-1:, n-1:] = Hx
        H = mdp.utils.mult(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = -D.prod()
    # Equivalent to mult(numx.diag(D), H) but faster
    H = (D*H.T).T
    return H

def norm2(v):
    """Compute the 2-norm for 1D arrays.
    norm2(v) = sqrt(sum(v_i^2))"""
    
    return numx.sqrt((v*v).sum())

def ordered_uniq(alist):
    """Return the elements in alist without repetitions.
    The order in the list is preserved.
    Implementation by Raymond Hettinger, 2002/03/17"""
    set_ = {}
    return [set_.setdefault(e, e) for e in alist if e not in set_]

def uniq(alist):
    """Return the elements in alist without repetitions.
    The order in the list is not preserved.
    Implementation by Raymond Hettinger, 2002/03/17"""
    set_ = {}
    map(set_.__setitem__, alist, [])
    return set_.keys()

def cov2(x, y):
    """Compute the covariance between 2D matrices x and y.
    Complies with the old scipy.cov function: different variables
    are on different columns."""

    mnx = x.mean(axis=0)
    mny = y.mean(axis=0)
    tlen = x.shape[0]
    return mdp.utils.mult(x.T, y)/(tlen-1) - numx.outer(mnx, mny)

def cov_maxima(cov):
    """Extract the maxima of a covariance matrix."""
    dim = cov.shape[0]
    maxs = []
    if dim >= 1:
        cov=abs(cov)
        glob_max_idx = (cov.argmax()/dim, cov.argmax()%dim)
        maxs.append(cov[glob_max_idx[0], glob_max_idx[1]])
        cov_reduce = cov.copy()
        cov_reduce = cov_reduce[numx.arange(dim) != glob_max_idx[0], :]
        cov_reduce = cov_reduce[:, numx.arange(dim) != glob_max_idx[1]]
        maxs.extend(cov_maxima(cov_reduce))
        return maxs
    else:
        return []
    

def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    
    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

def comb(N, k):
    """Return number of combinations of k objects from a set of N objects
    without repetitions, a.k.a. the binomial coefficient of N and k."""
    ret = 1
    for mlt in xrange(N, N-k, -1):
        ret *= mlt
    for dv in xrange(1, k+1):
        ret /= dv
    return ret


def get_dtypes(typecodes_key):
    """Return the list of dtypes corresponding to the set of
    typecodes defined in numpy.typecodes[typecodes_key].
    E.g., get_dtypes('Float') = [dtype('f'), dtype('d'), dtype('g')].
    """
    return [numx.dtype(c) for c in numx.typecodes[typecodes_key]]

# the following functions and classes were part of the scipy_emulation.py file

_type_keys = ['f', 'd', 'F', 'D']
_type_conv = {('f','d'): 'd', ('f','F'): 'F', ('f','D'): 'D',
              ('d','F'): 'D', ('d','D'): 'D',
              ('F','d'): 'D', ('F','D'): 'D'}

def _greatest_common_dtype(alist):
    """
    Apply conversion rules to find the common conversion type
    dtype 'd' is default for 'i' or unknown types
    (known types: 'f','d','F','D').
    """
    dtype = 'f'
    for array in alist:
        if array is None:
            continue
        tc = array.dtype.char
        if tc not in _type_keys:
            tc = 'd'
        transition = (dtype, tc)
        if transition in _type_conv:
            dtype = _type_conv[transition]
    return dtype

def _assert_eigenvalues_real_and_positive(w, dtype):
    tol = numx.finfo(dtype.type).eps * 100
    if abs(w.imag).max() > tol:
        err = "Some eigenvalues have significant imaginary part: %s " % str(w)
        raise SymeigException(err)
    #if w.real.min() < 0:
    #    err = "Got negative eigenvalues: %s" % str(w)
    #    raise SymeigException(err)
              
def wrap_eigh(A, B = None, eigenvectors = True, turbo = "on", range = None,
              type = 1, overwrite = False):
    """Wrapper for scipy.linalg.eigh for scipy version > 0.7"""
    args = {}
    args['a'] = A
    args['b'] = B
    args['eigvals_only'] = not eigenvectors
    args['overwrite_a'] = overwrite
    args['overwrite_b'] = overwrite
    if turbo == "on":
        args['turbo'] = True
    else:
        args['turbo'] = False
    args['type'] = type
    if range is not None:
        n = A.shape[0]
        lo, hi = range
        if lo < 1:
            lo = 1
        if lo > n:
            lo = n
        if hi > n:
            hi = n
        if lo > hi:
            lo, hi = hi, lo
        # in scipy.linalg.eigh the range starts from 0
        lo -= 1
        hi -= 1
        range = (lo, hi)
    args['eigvals'] = range
    try:
        return numx_linalg.eigh(**args)
    except numx_linalg.LinAlgError, exception:
        raise SymeigException(str(exception))

def _symeig_fake(A, B = None, eigenvectors = True, turbo = "on", range = None,
                 type = 1, overwrite = False):
    """Solve standard and generalized eigenvalue problem for symmetric
(hermitian) definite positive matrices.
This function is a wrapper of LinearAlgebra.eigenvectors or
numarray.linear_algebra.eigenvectors with an interface compatible with symeig.

    Syntax:

      w,Z = symeig(A) 
      w = symeig(A,eigenvectors=0)
      w,Z = symeig(A,range=(lo,hi))
      w,Z = symeig(A,B,range=(lo,hi))

    Inputs:

      A     -- An N x N matrix.
      B     -- An N x N matrix.
      eigenvectors -- if set return eigenvalues and eigenvectors, otherwise
                      only eigenvalues 
      turbo -- not implemented
      range -- the tuple (lo,hi) represent the indexes of the smallest and
               largest (in ascending order) eigenvalues to be returned.
               1 <= lo < hi <= N
               if range = None, returns all eigenvalues and eigenvectors. 
      type  -- not implemented, always solve A*x = (lambda)*B*x
      overwrite -- not implemented
      
    Outputs:

      w     -- (selected) eigenvalues in ascending order.
      Z     -- if range = None, Z contains the matrix of eigenvectors,
               normalized as follows:
                  Z^H * A * Z = lambda and Z^H * B * Z = I
               where ^H means conjugate transpose.
               if range, an N x M matrix containing the orthonormal
               eigenvectors of the matrix A corresponding to the selected
               eigenvalues, with the i-th column of Z holding the eigenvector
               associated with w[i]. The eigenvectors are normalized as above.
    """

    dtype = numx.dtype(_greatest_common_dtype([A, B]))
    try:
        if B is None:
            w, Z = numx_linalg.eigh(A)
        else:
            # make B the identity matrix
            wB, ZB = numx_linalg.eigh(B)
            _assert_eigenvalues_real_and_positive(wB, dtype)
            ZB = ZB.real / numx.sqrt(wB.real)
            # transform A in the new basis: A = ZB^T * A * ZB
            A = mdp.utils.mult(mdp.utils.mult(ZB.T, A), ZB)
            # diagonalize A
            w, ZA = numx_linalg.eigh(A)
            Z = mdp.utils.mult(ZB, ZA)
    except numx_linalg.LinAlgError, exception:
        raise SymeigException(str(exception))

    _assert_eigenvalues_real_and_positive(w, dtype)
    w = w.real
    Z = Z.real
    
    idx = w.argsort()
    w = w.take(idx)
    Z = Z.take(idx, axis=1)
    
    # sanitize range:
    n = A.shape[0]
    if range is not None:
        lo, hi = range
        if lo < 1:
            lo = 1
        if lo > n:
            lo = n
        if hi > n:
            hi = n
        if lo > hi:
            lo, hi = hi, lo
        
        Z = Z[:, lo-1:hi]
        w = w[lo-1:hi]

    # the final call to refcast is necessary because of a bug in the casting
    # behavior of Numeric and numarray: eigenvector does not wrap the LAPACK
    # single precision routines
    if eigenvectors:
        return mdp.utils.refcast(w, dtype), mdp.utils.refcast(Z, dtype)
    else:
        return mdp.utils.refcast(w, dtype)

def nongeneral_svd(A, range=None, **kwargs):
    """SVD routine for simple eigenvalue problem, API is compatible with
    symeig."""
    Z2, w, Z = mdp.utils.svd(A)
    # sort eigenvalues and corresponding eigenvectors
    idx = w.argsort()
    w = w.take(idx)
    Z = Z.take(idx, axis=1)
    # sort eigenvectors
    Z = (Z[-1::-1, -1::-1]).T
    if range is not None:
        lo, hi = range
        Z = Z[:, lo-1:hi]
        w = w[lo-1:hi]    
    return w, Z

def sqrtm(A):
    """This is a symmetric definite positive matrix sqrt function"""
    d, V = mdp.utils.symeig(A)
    return mdp.utils.mult(V, mult_diag(numx.sqrt(d), V.T))

# replication functions
def lrep(x, n):
    """Replicate x n-times on a new first dimension"""
    shp = [1]
    shp.extend(x.shape)
    return x.reshape(shp).repeat(n, axis=0)

def rrep(x, n):
    """Replicate x n-times on a new last dimension"""
    shp = x.shape + (1,)
    return x.reshape(shp).repeat(n, axis=-1)

def irep(x, n, dim):
    """Replicate x n-times on a new dimension dim-th dimension"""
    x_shape = x.shape
    shp = x_shape[:dim] + (1,) + x_shape[dim:]
    return x.reshape(shp).repeat(n, axis=dim)
# /replication functions


def orthogonal_permutations(a_dict):
    """
    Takes a dictionary with lists as keys and returns all permutations
    of these list elements in new dicts.
    
    This function is useful, when a method with several arguments
    shall be tested and all of the arguments can take several values.
    
    The order is not defined, therefore the elements should be
    orthogonal to each other.
    
    >>> for i in orthogonal_permutations({'a': [1,2,3], 'b': [4,5]}):
            print i
    {'a': 1, 'b': 4}
    {'a': 1, 'b': 5}
    {'a': 2, 'b': 4}
    {'a': 2, 'b': 5}
    {'a': 3, 'b': 4}
    {'a': 3, 'b': 5}
    """
    pool = dict(a_dict)
    args = []
    for func, all_args in pool.items():
        # check the size of the list in the second item of the tuple
        args_with_fun = [(func, arg) for arg in all_args]
        args.append(args_with_fun)
    for i in _product(args):
        yield dict(i)

def _product(iterable):
    # taken and adapted from itertools 2.6
    pools = tuple(iterable)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def weighted_choice(a_dict, is_normalised=False):
    """Returns a key from a dictionary based on the weight that the value suggests.
    If is_normalised=True, the sum of the weights must be one.
    
    TODO: It might be good to shuffle the order before using it to minimise possible degeneration.
    """
    if not is_normalised:
        d = a_dict.copy()
        s = sum(d.values())
        for key, val in d.items():
            d[key] = d[key] / s
    else:
        d = a_dict
    rand_num = random.random()
    total_rand = 0
    for key, val in d.items():
        total_rand += val
        if total_rand > rand_num:
            return key
    return None

def bool_to_sign(an_array):
    """Return -1 for each False; +1 for each True"""
    return numx.sign(an_array - 0.5)

def sign_to_bool(an_array, zero=True):
    """Return False for each negative value, else True"""
    if zero:
        return numx.array(an_array) >= 0
    else:
        return numx.array(an_array) > 0
