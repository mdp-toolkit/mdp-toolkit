import mdp
import warnings

# import numeric module (scipy, Numeric or numarray)
numx = mdp.numx
tr = numx.transpose

# precision warning parameters
_limits = { 'd' : 1e13, 'f' : 1e5}

def _check_roundoff(t, type):
    """Check if t is so large that t+1 == t up to 2 precision digits"""   
    if type.char in _limits:
        if int(t) >= _limits[type.char]:
            wr = 'You have summed %e entries in the covariance matrix.'%t+\
                 '\nAs you are using dtype \'%s\', you are '%type+\
                 'probably getting severe round off'+\
                 '\nerrors. See CovarianceMatrix docstring for more'+\
                 ' information.'
            warnings.warn(wr, mdp.MDPWarning)

class CovarianceMatrix(object):
    """This class stores an empirical covariance matrix that can be updated
    incrementally. A call to the function 'fix' returns the current state of
    the covariance matrix, the average and the number of observations, and
    resets the internal data.

    Note that the internal sum is a standard __add__ operation. We are not
    using any of the fancy sum algorithms to avoid round off errors when
    adding many numbers. If you want to contribute a CovarianceMatrix class
    that uses such algorithms we would be happy to include it in MDP.
    For a start see the Python recipe by Raymond Hettinger at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/393090
    For a review about floating point arithmetic and its pitfalls see
    http://docs.sun.com/source/806-3568/ncg_goldberg.html
    """

    def __init__(self, dtype = None, bias = False):
        """If dtype is not defined, it will be inherited from the first
        data bunch received by 'update'.
        All the matrices in this class are set up with the given dtype and
        no upcast is possible.
        If bias is True, the covariance matrix is normalized by dividing
        by T instead of the usual T-1.
        """
        
        if dtype is None:
            self._dtype = None
        else:
            self._dtype = numx.dtype(dtype)

        # covariance matrix, updated during the training phase
        self._cov_mtx = None
        # average, updated during the training phase
        self._avg = None
         # number of observation so far during the training phase
        self._tlen = 0

        self.bias = bias

    def _init_internals(self, x):
        """Inits some internals structures. The reason this is not done in
        the constructor is that we want to be able to derive the input
        dimension and the dtype directly from the data this class receives.
        """
        
        # init dtype
        if self._dtype is None:
            self._dtype = x.dtype
        dim = x.shape[1]
        self._input_dim = dim
        type = self._dtype
        # init covariance matrix
        self._cov_mtx = numx.zeros((dim,dim), type)
        # init average
        self._avg = numx.zeros(dim, type)

    def update(self, x):
        if self._cov_mtx is None:
            self._init_internals(x)
            
        #?? check the input dimension

        # cast input
        x = mdp.utils.refcast(x, self._dtype)
        
        # update the covariance matrix, the average and the number of
        # observations (try to do everything inplace)
        self._cov_mtx += mdp.utils.mult(tr(x), x)
        self._avg += numx.sum(x, 0)
        self._tlen += x.shape[0]

    def fix(self):
        """Returns a triple containing the covariance matrix, the average and
        the number of observations. The covariance matrix is then reset to
        a zero-state."""
        # local variables
        type = self._dtype
        tlen = self._tlen
        _check_roundoff(tlen, type)
        avg = self._avg
        cov_mtx = self._cov_mtx

        ##### fix the training variables
        # fix the covariance matrix (try to do everything inplace)
        avg_mtx = numx.outer(avg,avg)

        if self.bias:
            avg_mtx /= tlen*(tlen)
            cov_mtx /= tlen
        else:
            avg_mtx /= tlen*(tlen - 1)
            cov_mtx /= tlen - 1
        cov_mtx -= avg_mtx
        # fix the average
        avg /= tlen

        ##### clean up
        # covariance matrix, updated during the training phase
        self._cov_mtx = None
        # average, updated during the training phase
        self._avg = None
         # number of observation so far during the training phase
        self._tlen = 0

        return cov_mtx, avg, tlen


class DelayCovarianceMatrix(object):    
    """This class stores an empirical covariance matrix between the signal and
    time delayed signal that can be updated incrementally.

    Note that the internal sum is a standard __add__ operation. We are not
    using any of the fancy sum algorithms to avoid round off errors when
    adding many numbers. If you want to contribute a CovarianceMatrix class
    that uses such algorithms we would be happy to include it in MDP.
    For a start see the Python recipe by Raymond Hettinger at
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/393090
    For a review about floating point arithmetic and its pitfalls see
    http://docs.sun.com/source/806-3568/ncg_goldberg.html
    """

    def __init__(self, dt, dtype = None, bias = False):
        """dt is the time delay. If dt==0, DelayCovarianceMatrix equals
        CovarianceMatrix. If dtype is not defined, it will be inherited from
        the first data bunch received by 'update'.
        All the matrices in this class are set up with the given dtype and
        no upcast is possible.
        If bias is True, the covariance matrix is normalized by dividing
        by T instead of the usual T-1.
        """

        # time delay
        self._dt = int(dt)

        if dtype is None:
            self._dtype = None
        else:
            self._dtype = numx.dtype(dtype)

        # clean up variables to spare on space
        self._cov_mtx = None
        self._avg = None
        self._avg_dt = None
        self._tlen = 0

        self.bias = bias

    def _init_internals(self, x):
        """Inits some internals structures. The reason this is not done in
        the constructor is that we want to be able to derive the input
        dimension and the dtype directly from the data this class receives.
        """
        
        # init dtype
        if self._dtype is None:
            self._dtype = x.dtype
        dim = x.shape[1]
        self._input_dim = dim
        # init covariance matrix
        self._cov_mtx = numx.zeros((dim,dim), self._dtype)
        # init averages
        self._avg = numx.zeros(dim, self._dtype)
        self._avg_dt = numx.zeros(dim, self._dtype)

    def update(self, x):
        if self._cov_mtx is None:
            self._init_internals(x)

        # cast input
        x = mdp.utils.refcast(x, self._dtype)

        dt = self._dt

        # the number of data points in each block should be at least dt+1
        tlen = x.shape[0]
        if tlen < (dt+1):
            errstr = 'Block length is %d, should be at least %d.' % (tlen,dt+1)
            raise mdp.MDPException, errstr
        
        # update the covariance matrix, the average and the number of
        # observations (try to do everything inplace)
        self._cov_mtx += mdp.utils.mult(tr(x[:tlen-dt,:]), x[dt:tlen,:])
        totalsum = numx.sum(x, 0)
        self._avg += totalsum - numx.sum(x[tlen-dt:,:], 0)
        self._avg_dt += totalsum - numx.sum(x[:dt,:], 0)
        self._tlen += tlen-dt

    def fix(self, A=None):
        """The collected data is adjusted to compute the covariance matrix of
        the signal x(1)...x(N-dt) and the delayed signal x(dt)...x(N),
        which is defined as <(x(t)-<x(t)>)*(x(t+dt)-<x(t+dt)>)> .
        The function returns a tuple containing the covariance matrix,
        the average <x(t)> over the first N-dt points, the average of the
        delayed signal <x(t+dt)> and the number of observations. The internal
        data is then reset to a zero-state.
        
        If A is defined, the covariance matrix is transformed by the linear
        transformation Ax . E.g. to whiten the data, A is the whitening matrix.
        """
        
        # local variables
        type = self._dtype
        tlen = self._tlen
        _check_roundoff(tlen, type)
        avg = self._avg
        avg_dt = self._avg_dt
        cov_mtx = self._cov_mtx

        ##### fix the training variables
        # fix the covariance matrix (try to do everything inplace)
        avg_mtx = numx.outer(avg, avg_dt)
        avg_mtx /= tlen
                 
        cov_mtx -= avg_mtx
        if self.bias:
            cov_mtx /= tlen
        else:
            cov_mtx /= tlen - 1

        if A is not None:
            cov_mtx = mdp.utils.mult(A, mdp.utils.mult(cov_mtx, tr(A)))
        
        # fix the average
        avg /= tlen
        avg_dt /= tlen

        ##### clean up variables to spare on space
        self._cov_mtx = None
        self._avg = None
        self._avg_dt = None
        self._tlen = 0

        return cov_mtx, avg, avg_dt, tlen
