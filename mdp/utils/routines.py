import sys
import os
import cPickle
import tempfile
import mdp

# import numeric module (scipy, Numeric or numarray)
numx, numx_rand, numx_linalg  = mdp.numx, mdp.numx_rand, mdp.numx_linalg

class SymeigException(mdp.MDPException): pass

def timediff(data):
    """Returns the array of the time differences of data."""
    # this is the fastest way we found so far
    return data[1:]-data[:-1]

def refcast(array, dtype):
    """
    Cast the array to dtype only if necessary, otherwise return a reference.
    """
    if array.dtype == dtype:
        return array
    return array.astype(dtype)

def scast(scalar, dtype):
    """Convert a scalar in a 0D array of the given dtype."""
    return numx.array(scalar, dtype=dtype)

def rotate(mat, angle, columns = [0, 1], units = 'radians'):
    """
    Rotate in-place data matrix (NxM) in the plane defined by the columns=[i,j]
    when observation are stored on rows. Observation are rotated
    counterclockwise. This corresponds to the following matrix-multiplication
    for each data-point (unchanged elements omitted):

     [  cos(angle) -sin(angle)     [ x_i ]
        sin(angle)  cos(angle) ] * [ x_j ] 

    If M=2, columns=[0,1].
    """
    dtype = mat.dtype
    if units is 'degrees': angle = angle/180.*numx.pi
    cos_ = numx.cos(angle)
    sin_ = numx.sin(angle)
    [i,j] = columns
    col_i = mat[:,i] + 0.
    col_j = mat[:,j]
    mat[:,i] = cos_*col_i - sin_*col_j
    mat[:,j] = sin_*col_i + cos_*col_j

def hermitian(x):
    """Compute the Hermitian, i.e. conjugate transpose, of x."""
    return numx.conjugate(numx.transpose(x))

def symrand(dim_or_eigv, dtype="d"):
    """Return a random symmetric (Hermitian) matrix.
     if 'dim_or_eigv' is an integer N, return a NxN matrix.
     if 'dim_or_eigv' is  1-D real array 'a', return a matrix whose
                      eigenvalues are sort(a).
     """
    if isinstance(dim_or_eigv, int):
        dim = dim_or_eigv
        d = numx_rand.random(dim)
    elif isinstance(dim_or_eigv, numx.ndarray) and \
         len(numx.shape(dim_or_eigv)) == 1:
        dim = numx.shape(dim_or_eigv)[0]
        d = dim_or_eigv
    else:
        raise mdp.MDPException, "input type not supported."
    
    v = random_rot(dim, dtype=dtype)
    h = mdp.utils.mult(mdp.utils.mult(hermitian(v), mdp.numx.diag(d)), v)
    # to avoid roundoff errors, symmetrize the matrix (again)
    return refcast(0.5*(hermitian(h)+h), dtype)

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
        x[0] -= D[n-1]*mdp.numx.sqrt(mdp.numx.sum(x*x))
        # Householder transformation
        Hx = mdp.numx.eye(dim-n+1, dtype=dtype) \
             - 2.*mdp.numx.outer(x, x)/mdp.numx.sum(x*x)
        mat = mdp.numx.eye(dim, dtype=dtype)
        mat[n-1:,n-1:] = Hx
        H = mdp.utils.mult(H, mat)
    # Fix the last sign such that the determinant is 1
    D[n] = -mdp.numx.prod(D)
    # Equivalent to mult(numx.diag(D), H) but faster
    H = mdp.numx.transpose(D*mdp.numx.transpose(H))
    return H

def norm2(v):
    """Compute the 2-norm for 1D arrays.
    norm2(v) = sqrt(sum(v_i^2))"""
    
    return numx.sqrt(numx.sum(mdp.numx.squeeze(v*v)))

def ordered_uniq(alist):
    """Return the elements in alist without repetitions.
    The order in the list is preserved.
    Implementation by Raymond Hettinger, 2002/03/17"""
    set = {}
    return [set.setdefault(e,e) for e in alist if e not in set]

def uniq(alist):
    """Return the elements in alist without repetitions.
    The order in the list is not preserved.
    Implementation by Raymond Hettinger, 2002/03/17"""
    set = {}
    map(set.__setitem__, alist, [])
    return set.keys()

def cov2(x, y):
    """Compute the covariance between 2D matrices x and y.
    Complies with the old scipy.cov function: different variables
    are on different columns."""

    mnx = numx.mean(x, axis=0)
    mny = numx.mean(y, axis=0)
    tlen = x.shape[0]
    return mdp.utils.mult(numx.transpose(x), y)/(tlen-1) - numx.outer(mnx, mny)

class CrashRecoveryException(mdp.MDPException):
    """Class to handle crash recovery """
    def __init__(self, *args):
        """Allow crash recovery.
        Arguments: (error_string, crashing_obj, parent_exception)
        The crashing object is kept in self.crashing_obj
        The triggering parent exception is kept in self.parent_exception.
        """
        errstr = args[0]
        self.crashing_obj = args[1]
        self.parent_exception = args[2]
        # ?? python 2.5: super(CrashRecoveryException, self).__init__(errstr)
        mdp.MDPException.__init__(self, errstr)

    def dump(self, filename = None):
        """
        Save a pickle dump of the crashing object on filename.
        If filename is None, the crash dump is saved on a file created by
        the tempfile module.
        Return the filename.
        """
        if filename is None:
            (fd, filename) = tempfile.mkstemp(suffix=".pic",prefix="MDPcrash_")
            fl = os.fdopen(fd, 'w+b', -1)
        else:
            fl = file(filename, 'w+b',-1)
        cPickle.dump(self.crashing_obj,fl)
        fl.close()
        return filename


# the following functions and classes were part of the scipy_emulation.py file

_type_keys = ['f', 'd', 'F', 'D']
_type_conv = {('f','d'): 'd', ('f','F'): 'F', ('f','D'): 'D',
              ('d','F'): 'D', ('d','D'): 'D',
              ('F','d'): 'D', ('F','D'): 'D'}

def _greatest_common_dtype(alist):
    """
    Apply conversion rules to find the common conversion type
    Typecode 'd' is default for 'i' or unknown types
    (known types: 'f','d','F','D').
    """
    dtype = 'f'
    for array in alist:
        if array is None: continue
        tc = array.dtype.char
        if tc not in _type_keys: tc = 'd'
        transition = (dtype, tc)
        if transition in _type_conv:
            dtype = _type_conv[transition]
    return dtype

def _assert_eigenvalues_real(w, dtype):
    tol = numx.finfo(dtype.type).eps * 100
    if numx.amax(abs(w.imag)) > tol:
        raise SymeigException, \
              "Some eigenvalues have significant imaginary part"

def _symeig_fake(A, B = None, eigenvectors = 1, turbo = "on", range = None,
                 type = 1, overwrite = 0):
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
            w, Z = numx_linalg.eig(A)
        else:
            # make B the identity matrix
            wB, ZB = numx_linalg.eig(B)
            _assert_eigenvalues_real(wB, dtype)
            ZB = ZB.real / numx.sqrt(wB.real)
            # transform A in the new basis: A = ZB^T * A * ZB
            A = mdp.utils.mult(mdp.utils.mult(numx.transpose(ZB), A), ZB)
            # diagonalize A
            w, ZA = numx_linalg.eig(A)
            Z = mdp.utils.mult(ZB, ZA)
    except numx_linalg.LinAlgError, exception:
        raise SymeigException, str(exception)
    # workaround to bug in numpy 0.9.9
    except NameError, exception:
        if str(exception).strip() == \
               "NameError: global name 'Complex' is not defined":
            raise SymeigException, 'Complex eigenvalues'
        else:
            raise NameError, str(exception)

    _assert_eigenvalues_real(w, dtype)
    w = w.real
    Z = Z.real
        
    idx = numx.argsort(w)
    w = numx.take(w, idx)
    Z = numx.take(Z, idx, axis=1)
    
    if range is not None:
        lo, hi = range
        Z = Z[:, lo-1:hi]
        w = w[lo-1:hi]

    # the final call to refcast is necessary because of a bug in the casting
    # behavior of Numeric and numarray: eigenvector does not wrap the LAPACK
    # single precision routines
    if eigenvectors:
        return mdp.utils.refcast(w, dtype), mdp.utils.refcast(Z, dtype)
    else:
        return mdp.utils.refcast(w, dtype)

# Code found below this line is part, or derived from, SciPy 0.3.2.
# Copyright Notice:
#
##  Copyright (c) 2001, 2002 Enthought, Inc.
##
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##   a. Redistributions of source code must retain the above copyright notice,
##      this list of conditions and the following disclaimer.
##   b. Redistributions in binary form must reproduce the above copyright
##      notice, this list of conditions and the following disclaimer in the
##      documentation and/or other materials provided with the distribution.
##   c. Neither the name of the Enthought nor the names of its contributors
##      may be used to endorse or promote products derived from this software
##      without specific prior written permission.
##
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
## ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
## LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
## OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
## DAMAGE.
##

def sqrtm(A, disp=1):
    """This is a weak matrix sqrt function.
    It works only for symmetric matrices. disp : not implemented."""
    d, V = mdp.utils.symeig(A)
    D = numx.diag(numx.sqrt(d))
    return mdp.utils.mult(V, mdp.utils.mult(D, numx.transpose(V)))

# In file: scipy/common.py
def comb(N, k, exact=0):
    """Combinations of N things taken k at a time.

    Notes:
      - If k > N, N < 0, or k < 0, then a 0 is returned.

    Note:
    The exact=0 variant of scipy is not implemented.
    """
    if exact:
        if (k > N) or (N < 0) or (k < 0):
            return 0L
        N,k = map(long,(N,k))
        top = N
        val = 1L
        while (top > (N-k)):
            val *= top
            top -= 1
        n = 1L
        while (n < k+1L):
            val /= n
            n += 1
        return val
    else:
        raise NotImplementedError
