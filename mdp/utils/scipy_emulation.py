import mdp
import sys as _sys

# import numeric module (scipy, Numeric or numarray)
numx, numx_linalg = mdp.numx, mdp.numx_linalg
numx_rand, MDPException = mdp.numx_rand, mdp.MDPException

class SymeigException(MDPException): pass

class LeadingMinorException(SymeigException): pass

_type_keys = ['f', 'd', 'F', 'D']
_type_conv = {('f','d'): 'd', ('f','F'): 'F', ('f','D'): 'D',
              ('d','F'): 'D', ('d','D'): 'D',
              ('F','d'): 'D', ('F','D'): 'D'}

def _greatest_common_typecode(alist):
    """
    Apply conversion rules to find the common conversion type
    Typecode 'd' is default for 'i' or unknown types
    (known types: 'f','d','F','D').
    """
    typecode = 'f'
    for array in alist:
        if array is None: continue
        tc = array.typecode()
        if tc not in _type_keys: tc = 'd'
        transition = (typecode, tc)
        if transition in _type_conv:
            typecode = _type_conv[transition]
    return typecode

def _symeig_scipy(A, B = None, eigenvectors = 1, turbo = "on", range = None,
                  type = 1, overwrite = 0):
    """Solve standard and generalized eigenvalue problem for symmetric
(hermitian) definite positive matrices.
This function is a wrapper of scipy.eig with an interface compatible with
symeig.

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
      overwrite -- if 'overwrite' is set, computations are done inplace,
                   A and B are overwritten during calculation (you save
                   memory but loose the matrices).
      
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
    if eigenvectors:
        w, Z = numx_linalg.eig(A, B, left=0, right=1,
                               overwrite_a=overwrite, overwrite_b=overwrite)
        Z = Z
        w = numx.real(w)
        
        idx = numx.argsort(w)
        w = numx.take(w, idx)
        Z = numx.take(Z, idx, axis=1)

        if B is not None:
            # scipy assumes non-symmetric matrices, and normalizes the
            # generalized eigenvectors in another way. the following two
            # lines fix the difference.
            alpha = numx.diag(mdp.utils.mult(mdp.utils.mult(numx.transpose(Z),
                                                            B), Z))
            alpha = mdp.utils.refcast(alpha, B.typecode())
            Z = Z / numx.sqrt(alpha)
        
        if range is not None:
            lo, hi = range
            Z = Z[:, lo-1:hi]
            w = w[lo-1:hi]
        return w, Z
    else:
        w = numx_linalg.eig(A, B, left=0, right=0,
                            overwrite_a=overwrite, overwrite_b=overwrite)
        w = numx.sort(numx.real(w))
        if range is not None:
            lo, hi = range
            w = w[lo-1:hi]
        return w
        
def _symeig_dumb(A, B = None, eigenvectors = 1, turbo = "on", range = None,
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
    typecode = _greatest_common_typecode([A, B])
    if B is None:
        w, Z = numx_linalg.eigenvectors(A)
        Z = numx.transpose(Z)
        w = w.real
    else:
        # make B the identity matrix
        wB, ZB = numx_linalg.eigenvectors(B)
        ZB = numx.transpose(ZB) / numx.sqrt(wB.real)
        # transform A in the new basis: A = ZB^T * A * ZB
        A = mdp.utils.mult(mdp.utils.mult(numx.transpose(ZB), A), ZB)
        # diagonalize A
        w, ZA = numx_linalg.eigenvectors(A)
        Z = mdp.utils.mult(ZB, numx.transpose(ZA))
        w = w.real
        
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
        return mdp.utils.refcast(w, typecode), mdp.utils.refcast(Z, typecode)
    else:
        return mdp.utils.refcast(w, typecode)

def _scipy_normal(mean, std_dev, shape = ()):
    # interface as for RandomArray.normal
    return numx_rand.norm.rvs(size = shape, loc = mean, scale = std_dev)

# Code found below this line is part, or derived from, SciPy.
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

# In file: /home2/scipy/local/lib/python2.3/site-packages/scipy/common.py
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

# In file: scipy_base/shape_base.py
def squeeze(a):
    "Returns a with any ones from the shape of a removed"
    a = numx.asarray(a)
    b = numx.asarray(a.shape)
    val = numx.reshape(a, tuple(numx.compress(numx.not_equal(b, 1), b)))
    return val

# In file: scipy_base/matrix_base.py
# modified deprecated call to 'type'
# 6c6
# <     if type(M) == type('d'):
# ---
# >     if isinstance(M, str):
def eye(N, M=None, k=0, typecode='d'):
    """ eye returns a N-by-M matrix where the  k-th diagonal is all ones,
        and everything else is zeros.
    """
    if M is None: M = N
    if isinstance(M, str):
        typecode = M
        M = N
    m = numx.equal(numx.subtract.outer(numx.arange(N), numx.arange(M)),-k)
    if typecode is None:
        return m
    else:
        return m.astype(typecode)

# In file: scipy_base/matrix_base.py
def diag(v, k=0):
    """ returns the k-th diagonal if v is a matrix or returns a matrix
        with v as the k-th diagonal if v is a vector.
    """
    v = numx.asarray(v)
    s = v.shape
    if len(s)==1:
        n = s[0]+abs(k)
        if k > 0:
            v = numx.concatenate((numx.zeros(k, v.typecode()),v))
        elif k < 0:
            v = numx.concatenate((v,numx.zeros(-k, v.typecode())))
        return eye(n, k=k, typecode=v.typecode())*v
    elif len(s)==2:
        v = numx.add.reduce(eye(s[0], s[1], k=k, typecode=v.typecode())*v)
        if k > 0: return v[k:]
        elif k < 0: return v[:k]
        else: return v
    else:
        raise ValueError, "Input must be 1- or 2-D."

def sqrtm(A, disp=1):
    """This is a weak matrix sqrt function.
    It works only for symmetric matrices.
    disp : not implemented."""
    
    d, V = mdp.utils.symeig(A)
    D = diag(numx.sqrt(d))
    return mdp.utils.mult(V, mdp.utils.mult(D, numx.transpose(V)))

# In file: scipy/stats/stats.py
def _chk_asarray(a, axis):
    if axis is None:
        a = numx.ravel(a)
        outaxis = 0
    else:
        a = numx.asarray(a)
        outaxis = axis
    return a, outaxis

def _afun(fun, m, axis=-1):
    m, axis = _chk_asarray(m,axis)
    if len(m.shape)==0: m = numx.reshape(m,(1,))
    return fun.reduce(m, axis)

# In file: scipy_base/function_base.py
def amax(m, axis=-1):
    """Returns the maximum of m along dimension axis.
    """
    return _afun(numx.maximum, m, axis)

# In file: scipy_base/function_base.py
def amin(m,axis=-1):
    """Returns the minimum of m along dimension axis.
    """
    return _afun(numx.minimum, m, axis)

# In file: scipy/linalg/basic.py
def pinv(a, cond=1e-10):
    """ pinv(a, cond=None) -> a_pinv

    Compute generalized inverse of A using least-squares solver.
    """
    t = a.typecode()
    b = numx.identity(a.shape[0],t)
    return numx_linalg.linear_least_squares(a, b, rcond=cond)[0]

# In file: scipy_base/function_base.py
def linspace(start,stop,num=50,endpoint=1,retstep=0):
    """ Evenly spaced samples.
        Return num evenly spaced samples from start to stop.  If endpoint=1
        then last sample is stop. If retstep is 1 then return the step value
        used.
    """
    if num <= 0: return numx.array([])
    if endpoint:
        step = (stop-start)/float((num-1))
        y = numx.arange(0,num) * step + start        
    else:
        step = (stop-start)/float(num)
        y = numx.arange(0,num) * step + start
    if retstep:
        return y, step
    else:
        return y

# In file: scipy_base/shape_base.py
def atleast_2d(*arys):
    """ Force a sequence of arrays to each be at least 2D.

         Description:
            Force an array to each be at least 2D.  If the array
            is 0D or 1D, the array is converted to a single
            row of values.  Otherwise, the array is unaltered.
         Arguments:
            arys -- arrays to be converted to 2 or more dimensional array.
         Returns:
            input array converted to at least 2D array.
    """
    res = []
    for ary in arys:
        ary = numx.asarray(ary)
        if len(ary.shape) == 0: 
            ary = numx.array([ary[0]])
        if len(ary.shape) == 1: 
            result = ary[numx.NewAxis,:]
        else: 
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


# In file: scipy/stats/stats.py
def mean(a,axis=-1):
    """Returns the mean of m along the given dimension.
       If m is of integer type, returns a floating point answer.
    """
    a, axis = _chk_asarray(a, axis)
    return numx.add.reduce(a,axis)/float(a.shape[axis])

#In file: scipy_base/shape_base.py
def _expand_dims(a, axis):
    """Expand the shape of a by including NewAxis before given axis.
    """
    a = numx.asarray(a)
    shape = a.shape
    if axis < 0:
        axis = axis + len(shape) + 1
    a.shape = shape[:axis] + (1,) + shape[axis:]
    return a

# In file: scipy/stats/stats.py
def _ss(a, axis=-1):
    """
Squares each value in the passed array, adds these squares & returns
the result.  Axis can equal None (ravel array first), an integer
(the axis over which to operate), or a sequence (operate over
multiple axes).

Returns: sum-along-'axis' for (a*a)
"""
    a, axis = _chk_asarray(a, axis)
    return numx.sum(a*a,axis)

# In file: scipy/stats/stats.py
def var(a, axis=-1, bias=0):
    """
Returns the estimated population variance of the values in the passed
array (i.e., N-1).  Axis can equal None (ravel array first), or an
integer (the axis over which to operate).
"""
    a, axis = _chk_asarray(a, axis)
    mn = _expand_dims(mean(a,axis),axis)
    deviations = a - mn
    n = a.shape[axis]
    vals = _ss(deviations,axis)/(n-1.0)
    if bias:
        return vals * (n-1.0)/n
    else:
        return vals

# In file: scipy/stats/stats.py
def std (a, axis=-1, bias=0):
    """
Returns the estimated population standard deviation of the values in
the passed array (i.e., N-1).  Axis can equal None (ravel array
first), or an integer (the axis over which to operate).
"""
    return numx.sqrt(var(a,axis,bias))

# In file: scipy/stats/stats.py
def cov(m,y=None, rowvar=0, bias=0):
    """Estimate the covariance matrix.

    If m is a vector, return the variance.  For matrices where each row
    is an observation, and each column a variable, return the covariance
    matrix.  Note that in this case diag(cov(m)) is a vector of
    variances for each column.

    cov(m) is the same as cov(m, m)

    Normalization is by (N-1) where N is the number of observations
    (unbiased estimate).  If bias is 1 then normalization is by N.

    If rowvar is zero, then each row is a variables with
    observations in the columns.
    """
    m = numx.asarray(m)
    if y is None:
        y = m
    else:
        y = numx.asarray(y)
    if rowvar:
        m = numx.transpose(m)
        y = numx.transpose(y)
    N = m.shape[0]
    if (y.shape[0] != N):
        raise ValueError, "x and y must have the same number of observations."
    m = m - mean(m,axis=0)
    y = y - mean(y,axis=0)
    if bias:
        fact = N*1.0
    else:
        fact = N-1.0
    if y.typecode() in ['F', 'D']:
        yc = numx.conjugate(y)
    else:
        yc = y
    val = squeeze(mdp.utils.mult(numx.transpose(m), yc)) / fact
    return val


# In file: scipy_base/type_check.py
def iscomplexobj(x):
    return numx.asarray(x).typecode() in ['F', 'D']

# In file: scipy_test/testing.py
def assert_array_equal(x,y,err_msg=''):
    x,y = numx.asarray(x), numx.asarray(y)
    msg = '\nArrays are not equal'
    try:
        assert 0 in [len(numx.shape(x)),len(numx.shape(y))] \
               or (len(numx.shape(x))==len(numx.shape(y)) and \
                   numx.alltrue(numx.equal(numx.shape(x),numx.shape(y)))),\
                   msg + ' (shapes %s, %s mismatch):\n\t' \
                   % (numx.shape(x),numx.shape(y)) + err_msg
        reduced = numx.ravel(numx.equal(x,y))
        cond = numx.alltrue(reduced)
        if not cond:
            s1 = mdp.utils.array2string(x,precision=16)
            s2 = mdp.utils.array2string(y,precision=16)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg = msg + \
                  ' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s' % (match,
                                                                       s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        raise ValueError, msg

# In file: scipy_test/testing.py
def assert_array_almost_equal(x,y,decimal=6,err_msg=''):
    x = numx.asarray(x)
    y = numx.asarray(y)
    msg = '\nArrays are not almost equal'
    try:
        cond = numx.alltrue(numx.equal(numx.shape(x),numx.shape(y)))
        if not cond:
            msg=msg+' (shapes mismatch):\n\t'+ \
                 'Shape of array 1: %s\n\tShape of array 2: %s'%(numx.shape(x),
                                                                 numx.shape(y))
        assert cond, msg + '\n\t' + err_msg
        reduced = numx.ravel(numx.equal(\
            numx.less_equal(numx.around(abs(x-y),decimal),10.0**(-decimal)),1))
        cond = numx.alltrue(reduced)
        if not cond:
            s1 = mdp.utils.array2string(x,precision=decimal+1)
            s2 = mdp.utils.array2string(y,precision=decimal+1)
            if len(s1)>120: s1 = s1[:120] + '...'
            if len(s2)>120: s2 = s2[:120] + '...'
            match = 100-100.0*reduced.tolist().count(1)/len(reduced)
            msg=msg+' (mismatch %s%%):\n\tArray 1: %s\n\tArray 2: %s'%(match,
                                                                       s1,s2)
        assert cond,\
               msg + '\n\t' + err_msg
    except ValueError:
        print _sys.exc_value
        print numx.shape(x),numx.shape(y)
        print x, y
        raise ValueError, 'arrays are not almost equal'

# In file: scipy_test/testing.py
def assert_equal(actual,desired,err_msg='',verbose=1):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
    """
    msg = '\nItems are not equal:\n' + err_msg
    try:
        if ( verbose and len(repr(desired)) < 100 and len(repr(actual)) ):
            msg =  msg \
                 + 'DESIRED: ' + repr(desired) \
                 + '\nACTUAL: ' + repr(actual)
    except:
        msg =  msg \
             + 'DESIRED: ' + repr(desired) \
             + '\nACTUAL: ' + repr(actual)
    assert desired == actual, msg

# In file: scipy_test/testing.py
def assert_almost_equal(actual,desired,decimal=7,err_msg='',verbose=1):
    """ Raise an assertion if two items are not
        equal.  I think this should be part of unittest.py
    """
    msg = '\nItems are not equal:\n' + err_msg
    try:
        if ( verbose and len(repr(desired)) < 100 and len(repr(actual)) ):
            msg =  msg \
                 + 'DESIRED: ' + repr(desired) \
                 + '\nACTUAL: ' + repr(actual)
    except:
        msg =  msg \
             + 'DESIRED: ' + repr(desired) \
             + '\nACTUAL: ' + repr(actual)
    assert round(abs(desired - actual),decimal) == 0, msg


