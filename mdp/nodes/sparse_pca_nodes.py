import mdp
from mdp import numx, Node, NodeException, MDPWarning
from mdp.utils import mult
import warnings as _warnings
#from mdp.numx import sparse # XXX check that numx is actually scipy

def _check_roundoff(t, dtype):
    """Check if t is so large that t+1 == t up to 2 precision digits"""
    # limit precision
    limit = 10.**(numx.finfo(dtype).precision-2)
    if int(t) >= limit:
        wr = ('You have summed %e entries in the covariance matrix.'
              '\nAs you are using dtype \'%s\', you are '
              'probably getting severe round off'
              '\nerrors. See CovarianceMatrix docstring for more'
              ' information.' % (t, dtype.name))
        warnings.warn(wr, mdp.MDPWarning)

class SparseCovarianceMatrix(object):
    """This class stores an empirical covariance matrix that can be updated
    incrementally. A call to the 'fix' method returns the current state of
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

    def __init__(self, dtype=None, bias=False):
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
        self._input_dim = None  # will be set in _init_internals
        # covariance matrix, updated during the training phase
        self._cov_mtx = None
        # average, updated during the training phase
        self._avg = None
        # number of observation so far during the training phase
        self._tlen = 0

        self.bias = bias

    def _init_internals(self, x):
        """Init the internal structures.

        The reason this is not done in the constructor is that we want to be
        able to derive the input dimension and the dtype directly from the
        data this class receives.
        """
        # init dtype
        if self._dtype is None:
            self._dtype = x.dtype
        dim = x.shape[1]
        self._input_dim = dim
        type_ = self._dtype
        # init covariance matrix
        self._cov_mtx = numx.sparse.csr_matrix((dim, dim), dtype=type_)
        # init average
        self._avg = numx.sparse.csr_matrix((1, dim), dtype=type_)

    def update(self, x):
        """Update internal structures.

        Note that no consistency checks are performed on the data (this is
        typically done in the enclosing node).
        """
        if self._cov_mtx is None:
            self._init_internals(x)
        # cast input
        x = mdp.utils.refcast(x, self._dtype)
        # update the covariance matrix, the average and the number of
        # observations (try to do everything inplace)
        self._cov_mtx = self._cov_mtx + mdp.utils.mult(x.T, x)
        self._avg = self._avg + x.sum(axis=0)
        self._tlen += x.shape[0]

    def fix(self):
        """Returns a triple containing the covariance matrix, the average and
        the number of observations. The covariance matrix is then reset to
        a zero-state."""
        # local variables
        type_ = self._dtype
        tlen = self._tlen
        _check_roundoff(tlen, type_)
        avg = self._avg
        cov_mtx = self._cov_mtx

        ##### fix the training variables
        # fix the covariance matrix (try to do everything inplace)
        avg_mtx = mdp.utils.mult(avg.T, avg)

        if self.bias:
            avg_mtx /= tlen*(tlen)
            cov_mtx /= tlen
        else:
            avg_mtx /= tlen*(tlen - 1)
            cov_mtx /= tlen - 1
        cov_mtx = cov_mtx - avg_mtx
        # fix the average
        avg = avg/tlen

        ##### clean up
        # covariance matrix, updated during the training phase
        self._cov_mtx = None
        # average, updated during the training phase
        self._avg = None
        # number of observation so far during the training phase
        self._tlen = 0

        return cov_mtx, avg, tlen

class SparsePCANode(Node):
    """Filter the input data through the most significatives of its
    principal components.
    
    Internal variables of interest:
    self.avg -- Mean of the input data (available after training)
    self.v -- Transposed of the projection matrix (available after training)
    self.d -- Variance corresponding to the PCA components
              (eigenvalues of the covariance matrix)
    self.explained_variance -- When output_dim has been specified as a fraction
                               of the total variance, this is the fraction
                               of the total variance that is actually explained
    
    
    More information about Principal Component Analysis, a.k.a. discrete
    Karhunen-Loeve transform can be found among others in
    I.T. Jolliffe, Principal Component Analysis, Springer-Verlag (1986).
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 reduce=False, var_rel=1E-12, var_abs=1E-15,
                 var_part=None):
        """The number of principal components to be kept can be specified as
        'output_dim' directly (e.g. 'output_dim=10' means 10 components
        are kept) or by the fraction of variance to be explained
        (e.g. 'output_dim=0.95' means that as many components as necessary
        will be kept in order to explain 95% of the input variance).

        Other Keyword Arguments:

        reduce -- Keep only those principal components which have a variance
                  larger than 'var_abs' and a variance relative to the
                  first principal component larger than 'var_rel' and a
                  variance relative to total variance larger than 'var_part'
                  (set var_part to None or 0 for no filtering).
                  Note: when the 'reduce' switch is enabled, the actual number
                  of principal components (self.output_dim) may be different
                  from that set when creating the instance.
        """
        # this must occur *before* calling super!
        self.desired_variance = None
        super(SparsePCANode, self).__init__(input_dim, output_dim, dtype)
        self.var_abs = var_abs
        self.var_rel = var_rel
        self.var_part = var_part
        self.reduce = reduce
        # empirical covariance matrix, updated during the training phase
        self._cov_mtx = SparseCovarianceMatrix(dtype)
        # attributes that defined in stop_training
        self.d = None  # eigenvalues  
        self.v = None  # eigenvectors, first index for coordinates
        self.total_variance = None
        self.tlen = None
        self.avg = None
        self.explained_variance = None
        
    def _set_output_dim(self, n):
        if n <= 1 and isinstance(n, float):
            # set the output dim after training, when the variances are known 
            self.desired_variance = n
        else:
            self._output_dim = n
        
    def _check_output(self, y):
        # check output rank
        if not y.ndim == 2:
            error_str = "y has rank %d, should be 2" % (y.ndim)
            raise NodeException(error_str)

        if y.shape[1] == 0 or y.shape[1] > self.output_dim:
            error_str = ("y has dimension %d" 
                         ", should be 0<y<=%d" % (y.shape[1], self.output_dim))
            raise NodeException(error_str)

    def _get_supported_dtypes(self):
        return ['float32', 'float64']
    
    def get_explained_variance(self):
        """Return the fraction of the original variance that can be
        explained by self._output_dim PCA components.
        If for example output_dim has been set to 0.95, the explained
        variance could be something like 0.958...
        Note that if output_dim was explicitly set to be a fixed number
        of components, there is no way to calculate the explained variance.
        """
        return self.explained_variance
    
    def _train(self, x):
        # update the covariance matrix
        self._cov_mtx.update(x)

    def _adjust_output_dim(self):
        """Return the eigenvector range and set the output dim if required.
        
        This is used if the output dimensions is smaller than the input
        dimension (so only the larger eigenvectors have to be kept). 
        """
        # if the number of principal components to keep is not specified,
        # keep all components
        if self.desired_variance is None and self.output_dim is None:
            self.output_dim = self.input_dim
            return None

        ## define the range of eigenvalues to compute
        # if the number of principal components to keep has been
        # specified directly
        if self.output_dim is not None and self.output_dim >= 1:
            # (eigenvalues sorted in ascending order)
            return (self.input_dim - self.output_dim + 1,
                   self.input_dim)
        # otherwise, the number of principal components to keep has been
        # specified by the fraction of variance to be explained
        else:
            return None
        
    def _stop_training(self, debug=False):
        """Stop the training phase.

        Keyword arguments:

        debug=True     if stop_training fails because of singular cov
                       matrices, the singular matrices itselves are stored in
                       self.cov_mtx and self.dcov_mtx to be examined.
        """
        # request the covariance matrix and clean up
        self.cov_mtx, avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx

        # range for the eigenvalues
        rng = self._adjust_output_dim()
        
        # if we have more variables then observations we are bound to fail here
        # suggest to use the NIPALSNode instead.
        if debug and self.tlen < self.input_dim:
            wrn = ('The number of observations (%d) '
                   'is larger than the number of input variables '
                   '(%d). You may want to use ' 
                   'the NIPALSNode instead.' % (self.tlen, self.input_dim))
            _warnings.warn(wrn, MDPWarning)

        # total variance can be computed at this point:
        # note that vartot == d.sum()
        vartot = numx.diag(self.cov_mtx).sum()
        
        ## compute and sort the eigenvalues
        # compute the eigenvectors of the covariance matrix (inplace)
        # (eigenvalues sorted in ascending order)
        try:
            d, v = mdp.numx.sparse.svd(self.cov_mtx)
            self._symeig(self.cov_mtx, k=rng, overwrite=(not debug))
            # if reduce=False and svd=False. we should check for
            # negative eigenvalues and fail
            if not (self.reduce or self.svd or (self.desired_variance is
                                                not None)):
                if d.min() < 0:
                    raise NodeException("Got negative eigenvalues: "
                                        "%s.\n"
                                        "You may either set output_dim to be"
                                        " smaller, or set reduce=True and/or "
                                        "svd=True" % str(d))
        except SymeigException, exception:
            err = str(exception)+("\nCovariance matrix may be singular."
                                  "Try setting svd=True.")
            raise NodeException(err)
                  
        # delete covariance matrix if no exception occurred
        if not debug:
            del self.cov_mtx
        
        # sort by descending order
        d = numx.take(d, range(d.shape[0]-1, -1, -1))
        v = v[:, ::-1]

        if self.desired_variance is not None:
            # throw away immediately negative eigenvalues
            d = d[ d > 0 ]
            # the number of principal components to keep has
            # been specified by the fraction of variance to be explained
            varcum = (d / vartot).cumsum(axis=0)
            # select only the relevant eigenvalues
            # number of relevant eigenvalues
            neigval = varcum.searchsorted(self.desired_variance) + 1.
            #self.explained_variance = varcum[neigval-1]
            # cut
            d = d[0:neigval]
            v = v[:, 0:neigval]
            # define the new output dimension
            self.output_dim = int(neigval)

        # automatic dimensionality reduction
        if self.reduce:
            # remove entries that are smaller then var_abs and
            # smaller then var_rel relative to the maximum
            d = d[ d > self.var_abs ]
            d = d[ d / d.max() > self.var_rel ]
            
            # filter for variance relative to total variance
            if self.var_part:
                d = d[ d / vartot > self.var_part ]
            
            v = v[:, 0:d.shape[0]]
            self._output_dim = d.shape[0]
            
        # set explained variance
        self.explained_variance = d.sum() / vartot
        
        # store the eigenvalues
        self.d = d
        # store the eigenvectors
        self.v = v
        # store the total variance
        self.total_variance = vartot

    def get_projmatrix(self, transposed=1):
        """Return the projection matrix."""
        self._if_training_stop_training()
        if transposed:
            return self.v
        return self.v.T

    def get_recmatrix(self, transposed=1):
        """Return the back-projection matrix (i.e. the reconstruction matrix).
        """
        self._if_training_stop_training()
        if transposed:
            return self.v.T
        return self.v

    def _execute(self, x, n=None):
        """Project the input on the first 'n' principal components.
        If 'n' is not set, use all available components."""
        if n is not None:
            return mult(x-self.avg, self.v[:, :n])
        return mult(x-self.avg, self.v)

    def _inverse(self, y, n=None):
        """Project 'y' to the input space using the first 'n' components.
        If 'n' is not set, use all available components."""
        if n is None:
            n = y.shape[1]
        if n > self.output_dim:
            error_str = ("y has dimension %d,"
                         " should be at most %d" % (n, self.output_dim))
            raise NodeException(error_str)
        
        v = self.get_recmatrix()
        if n is not None:
            return mult(y, v[:n, :]) + self.avg
        return mult(y, v) + self.avg


class SparseWhiteningNode(SparsePCANode):
    """'Whiten' the input data by filtering it through the most
    significatives of its principal components. All output
    signals have zero mean, unit variance and are decorrelated.
    
    Internal variables of interest:
    self.avg -- Mean of the input data (available after training)
    self.v -- Transpose of the projection matrix (available after training)
    self.d -- Variance corresponding to the PCA components
              (eigenvalues of the covariance matrix).
    self.explained_variance -- When output_dim has been specified as a fraction
                               of the total variance, this is the fraction
                               of the total variance that is actually explained
   
    """
    
    def _stop_training(self, debug=False):
        super(SparseWhiteningNode, self)._stop_training(debug)

        ##### whiten the filters
        # self.v is now the _whitening_ matrix
        self.v = self.v / numx.sqrt(self.d)

    def get_eigenvectors(self):
        """Return the eigenvectors of the covariance matrix."""
        self._if_training_stop_training()
        return numx.sqrt(self.d)*self.v

    def get_recmatrix(self, transposed=1):
        """Return the back-projection matrix (i.e. the reconstruction matrix).
        """
        self._if_training_stop_training()
        v_inverse = self.v*self.d
        if transposed:
            return v_inverse.T
        return v_inverse
