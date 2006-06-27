## Automatically adapted for numpy Jun 26, 2006 by 

import sys
import os
import cPickle
import tempfile
import mdp

# import numeric module (scipy, Numeric or numarray)
numx, numx_rand, numx_linalg  = mdp.numx, mdp.numx_rand, mdp.numx_linalg

def timediff(data):
    """Returns the array of the time differences of data."""
    # this is the fastest way we found so far
    return data[1:]-data[:-1]

def refcast(array, typecode):
    """
    Cast the array to typecode only if necessary, otherwise return a reference.
    """
    if array.dtype.char==typecode:
        return array
    return array.astype(typecode)

def scast(scalar, typecode):
    """Convert a scalar in a 0D array of the given typecode."""
    return numx.array(scalar, dtype=typecode)

def _gemm_matmult(a,b, alpha=1.0, beta=0.0, c=None, trans_a=0, trans_b=0):
    """Return alpha*(a*b) + beta*c.
    a,b,c : matrices
    alpha, beta: scalars
    trans_a: 0 (a not transposed), 1 (a transposed), 2 (a conjugate transposed)
    trans_b: 0 (b not transposed), 1 (b transposed), 2 (b conjugate transposed)
    """
    typecode = mat.dtype.char
    if a.flags.contiguous and b.flags.contiguous and not trans_a and not trans_b:
        mat = numx.dot(a, b)
        if alpha != 1:
            mat *= scast(alpha, typecode)
        if beta != 0:
            mat += scast(beta*c, typecode)
        return mat
    if c is not None:
        gemm, = numx_linalg.get_blas_funcs(('gemm',),(a,b,c))
    else:
        gemm,=  numx_linalg.get_blas_funcs(('gemm',),(a,b))

    return gemm(alpha, a, b, beta, c, trans_a, trans_b)

def _matmult(a,b):
    """Return matrix multiplication between 2-dimensional matrices a and b."""

    if (numx.rank(a)!=2 or numx.rank(b)!=2) or \
           (a.flags.contiguous and b.flags.contiguous):
        return numx.dot(a,b)
    else:
        if a.shape[1] != b.shape[0]:
            err_str = "matrices are not aligned. shape(a)=%s, shape(b)=%s"\
                      %(str(a.shape),str(b.shape))
            raise mdp.MDPException, err_str
        gemm, = numx_linalg.get_blas_funcs(('gemm',),(a,b))
        return gemm(1.0, a, b, 0., None, 0, 0)

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
    typecode = mat.dtype.char
    if units is 'degrees': angle = angle/180.*numx.pi
    cos_ = scast(numx.cos(angle), typecode)
    sin_ = scast(numx.sin(angle), typecode)
    [i,j] = columns
    col_i = mat[:,i] + scast(0, typecode)
    col_j = mat[:,j]
    mat[:,i] = cos_*col_i - sin_*col_j
    mat[:,j] = sin_*col_i + cos_*col_j

def hermitian(x):
    """Compute the Hermitian, i.e. conjugate transpose, of x."""
    return numx.conjugate(numx.transpose(x))

def symrand(dim_or_eigv, typecode="d"):
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
    
    v = random_rot(dim, typecode=typecode)
    h = mdp.utils.mult(mdp.utils.mult(hermitian(v), mdp.utils.diag(d)), v)
    # to avoid roundoff errors, symmetrize the matrix (again)
    return refcast(0.5*(hermitian(h)+h), typecode)

def random_rot(dim, typecode='d'):
    """Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization"""
    H = mdp.numx.eye(dim, dtype=typecode)
    D = mdp.numx.ones((dim,), dtype=typecode)
    for n in range(1, dim):
        x = mdp.numx_rand.normal(size=(dim-n+1,)).astype(typecode)
        D[n-1] = mdp.numx.sign(x[0])
        x[0] -= D[n-1]*mdp.numx.sqrt(mdp.numx.sum(x*x))
        # Householder transformation
        Hx = mdp.numx.eye(dim-n+1, dtype=typecode) \
             - 2.*mdp.numx.outer(x, x)/mdp.numx.sum(x*x)
        mat = mdp.numx.eye(dim, dtype=typecode)
        mat[n-1:,n-1:] = Hx
        H = mdp.utils.mult(H, mat)
    # Fix the last sign such that the determinant is 1
    D[n] = -mdp.numx.prod(D)
    # Equivalent to mult(numx.diag(D), H) but faster
    H = mdp.numx.transpose(D*mdp.numx.transpose(H))
    return H

class ProgressBar(object):
    """A text-mode fully configurable progress bar.
       Note: remember that ProgressBar flushes sys.stdout by
       default each time you update it. If you need to rely on
       buffered stdout than set flush = 0.
    """

    def __init__(self,minimum,maximum,width = None,delimiters = "[]",\
                 char1 = "=",char2 = ">",char3 = "." , indent = 0, flush = 1):
        """
        (minimum-maximum) - the total number of iterations for which you
                            want to show a progress bar.
        width             - the total width of the progress bar in characters
                            (default is the teminal widthin UNIX and
                             79 characters in Windows).
        The default progress bar looks like this:
        [===========60%===>.........]
        Progress bar delimiters and chars can be customized. 
        """
        # bar is [===>.....20%.....]
        self.flush = flush
        self.delimiters = delimiters
        self.bar = delimiters   # This holds the progress bar string
        self.min = minimum
        self.max = maximum
        self.span = maximum - minimum
        self.char1 = char1
        self.char2 = char2
        self.char3= char3
        self.indent = indent
        self.where = 0       # When where == max, we are 100% done
        if not width:
            self.width = self.get_termsize()[1]-2  # terminal width-2 
        else:
            self.width = width
        self.width = self.width - indent
        
    def update(self, where = 0):
        if where < self.min: where = self.min
        if where > self.max: where = self.max
        self.where = where

        # Figure out the new percent done
        if self.span == 0:
            percent = 1.
        else:
            percent = (self.where - self.min)/ float(self.span)

        # Figure out how many hash bars the percentage should be
        actwidth = self.width - 2
        symbols = int(round(percent * actwidth))

        # build a progress bar with hashes and spaces
        self.bar =  self.delimiters[0]+self.char1*(symbols-1)+\
                   self.char2+self.char3*(actwidth-symbols)+self.delimiters[1]

        # figure out where to put the percentage, roughly centered
        percent = int(round(percent*100))
        percentstring = str(percent) + "%"
        placepercent = (len(self.bar) / 2) - len(percentstring) + 2
        
        # slice the percentage into the bar
        self.bar = self.bar[0:placepercent] + percentstring +\
                   self.bar[placepercent+len(percentstring):]
        print "\r" + " "*self.indent + str(self.bar),
        if self.flush:
            sys.stdout.flush()
        
    def get_termsize(self):
        try:
            # this works on unix machines
            import struct, fcntl, termios
            height, width = struct.unpack("hhhh", \
                     fcntl.ioctl(0,termios.TIOCGWINSZ, "\000"*8))[0:2]
            if not (height and width): height, width = 24, 79
        except ImportError:
            # for windows machins, use default values
            # Does anyone know how to get the console size under windows?
            # and what about MacOsX?
            height, width = 24, 79
        return height, width    

def norm2(v):
    """Compute the 2-norm for 1D arrays.
    norm2(v) = sqrt(sum(v_i^2))"""
    
    return numx.sqrt(numx.sum(mdp.utils.squeeze(v*v)))

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

