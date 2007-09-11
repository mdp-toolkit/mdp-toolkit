from mdp import numx, Node, \
     NodeException, TrainingFinishedException
from mdp.utils import mult, symeig, nongeneral_svd, CovarianceMatrix, \
                      SymeigException #, LeadingMinorException

class PCANode(Node):
    """Filter the input data throug the most significatives of its
    principal components.
    
    Internal variables of interest:
    self.avg -- Mean of the input data (available after training)
    self.v -- Transposed of the projection matrix (available after training)
    self.d -- Variance corresponding to the PCA components
              (eigenvalues of the covariance matrix).
    
    More information about Principal Component Analysis, a.k.a. discrete
    Karhunen-Loeve transform can be found among others in
    I.T. Jolliffe, Principal Component Analysis, Springer-Verlag (1986)."""
    
    def __init__(self, input_dim = None, output_dim = None, dtype = None):
        """The number of principal components to be kept can be specified as
        'output_dim' directly (e.g. 'output_dim=10' means 10 components
        are kept) or by the fraction of variance to be explained
        (e.g. 'output_dim=0.95' means that as many components as necessary
        will be kept in order to explain 95% of the input variance)."""
        if output_dim <= 1 and isinstance(output_dim, float):
            self.desired_variance = output_dim
            output_dim = None
        else:
            self.desired_variance = None
        
        super(PCANode, self).__init__(input_dim, output_dim, dtype)

        # routine for eigenvalues
        self._symeig = symeig
        # empirical covariance matrix, updated during the training phase
        self._cov_mtx = CovarianceMatrix(dtype)
        
    def _check_output(self, y):
        # check output rank
        if not y.ndim == 2:
            error_str = "y has rank %d, should be 2"\
                        %(y.ndim)
            raise NodeException, error_str

        if y.shape[1]==0 or y.shape[1]>self.output_dim:
            error_str = "y has dimension %d, should be 0<y<=%d" \
                        % (y.shape[1], self.output_dim)
            raise NodeException, error_str

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

    def stop_training(self, debug=False):
        """Stop the training phase.

        Keyword arguments:

        debug=True     if stop_training fails because of singular cov
                       matrices, the singular matrices itselves are stored in
                       self.cov_mtx and self.dcov_mtx to be examined.
        """
        return super(PCANode, self).stop_training(debug=debug)
        
    def _stop_training(self, debug=False):
        ##### request the covariance matrix and clean up
        self.cov_mtx, avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx
        
        # this is a bit counterintuitive, as it reshapes the average vector to
        # be a matrix. in this way, however, we spare the reshape
        # operation every time that 'execute' is called.
        self.avg = avg.reshape(1, avg.shape[0])

        ##### compute the principal components
        # if the number of principal components to keep is not specified,
        # keep all components
        if self.desired_variance is None and self.output_dim is None:
            self.output_dim = self.input_dim

        ## define the range of eigenvalues to compute
        # if the number of principal components to keep has been
        # specified directly
        if self.output_dim >= 1:
            # (eigenvalues sorted in ascending order)
            rng = (self.input_dim-self.output_dim+1, self.input_dim)
        # otherwise, the number of principal components to keep has been
        # specified by the fraction of variance to be explained
        else:
            rng = None

        ## compute and sort the eigenvalues
        # compute the eigenvectors of the covariance matrix (inplace)
        # (eigenvalues sorted in ascending order)
        try:
            d, v = self._symeig(self.cov_mtx, range=rng, overwrite=(not debug))
        except SymeigException, exception:
            errstr = str(exception)+"\n Covariance matrix may be singular."
            raise NodeException,errstr
                  
        # delete covariance matrix if no exception occurred
        del self.cov_mtx
        
        # sort by descending order
        d = numx.take(d, range(d.shape[0]-1,-1,-1))
        v = v[:,::-1]

        ## compute the explained variance
        # if the number of principal components to keep has been
        # specified directly
        if self.output_dim >= 1:
            # there is no way to tell what the explained variance is, since we
            # didn't compute all eigenvalues
            self.explained_variance = None
        elif self.output_dim == self.input_dim:
            # explained variance is 100%
            self.explained_variance = 1.
        else:
            # otherwise, the number of principal components to keep has
            # been specified by the fraction of variance to be explained
            #
            # total variance
            vartot = d.sum()
            # cumulative variance (percent)
            varcum = (d/vartot).cumsum(axis=0)
            # select only the relevant eigenvalues
            # number of relevant eigenvalues
            neigval = varcum.searchsorted(self.desired_variance) + 1.
            self.explained_variance = varcum[neigval]
            # cut
            d = d[0:neigval]
            v = v[:,0:neigval]
            # define the new output dimension
            self.output_dim = neigval
            
        ## store the eigenvalues
        self.d = d

        ## store the eigenvectors
        self.v = v

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

    def _execute(self, x, n = None):
        if n is not None:
            return mult(x-self.avg, self.v[:,:n])
        return mult(x-self.avg, self.v)

    def execute(self, x, n=None):
        """Project the input on the first 'n' principal components.
        If 'n' is not set, use all available components."""
        return super(PCANode, self).execute(x, n)

    def _inverse(self, y, n = None):
        """Project 'y' to the input space using the first 'n' components.
        If 'n' is not set, use all available components."""
        if n is None:
            n = y.shape[1]
        if n > self.output_dim:
            error_str = "y has dimension %d, should be at most %d" \
                        %(n, self.output_dim)
            raise NodeException, error_str
        
        v = self.get_recmatrix()
        return mult(y, v[:n,:])+self.avg

    def inverse(self, y, n=None):
        """Project 'y' to the input space using the first 'n' components.
        If 'n' is not set, use all available components."""
        return super(PCANode, self).inverse(y, n)


class WhiteningNode(PCANode):
    """'Whiten' the input data by filtering it through the most
    significatives of its principal components. All output
    signals have zero mean, unit variance and are decorrelated.
    
    Internal variables of interest:
    self.avg -- Mean of the input data (available after training)
    self.v -- Transpose of the projection matrix (available after training)
    self.d -- Variance corresponding to the PCA components
              (eigenvalues of the covariance matrix).
    """
    
    def _stop_training(self, debug=False):
        super(WhiteningNode, self)._stop_training(debug)

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
    
class PCASVDNode(PCANode):
    """Perform PCA (see docs for PCANode) using Singular Value Decomposition
    instead of the standard Eigenvalue Problem Solver. Use it when PCANode
    complains about singular covariance matrices or when you know that the
    covariance matrix of your data has zero eigenvalues."""

    def __init__(self, input_dim = None, output_dim = None, dtype = None,
                 reduce=False, eps_abs=1E-15, eps_rel=1E-15):
        
        """Keyword Arguments

        reduce -- if True automatically delete dimensions"""    
        super(PCASVDNode, self).__init__(input_dim, output_dim, dtype)
        self._symeig = nongeneral_svd
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.reduce = reduce

    def _stop_training(self, debug=False):
        super(PCASVDNode, self)._stop_training(debug)
        if self.reduce:
            # remove entries that are smaller then eps_abs and
            # smaller then eps_rel relative to the maximum
            self.d = self.d[ self.d > self.eps_abs ]
            self.d = self.d[ self.d/self.d.max() > self.eps_rel ]
            self.v = self.v[:,0:self.d.shape[0]]
            self._output_dim = self.d.shape[0]
    
class WhiteningSVDNode(PCASVDNode):
    """Perform Whitening (see docs for WhiteningNode) using Singular Value
    Decomposition instead of the standard Eigenvalue Problem Solver. Use it
    when WhiteningNode complains about singular covariance matrices or when
    you know that the covariance matrix of your data has zero eigenvalues."""

    def _stop_training(self, debug=False):
        super(WhiteningSVDNode, self)._stop_training(debug)

        ##### whiten the filters
        # self.v is now the _whitening_ matrix
        # automatically correct for negative eigenvalues
        self.v = self.v / numx.sqrt(abs(self.d))

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
