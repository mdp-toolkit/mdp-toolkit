from mdp import numx, SignalNode, SignalNodeException
from mdp.utils import mult, symeig, LeadingMinorException
from lcov import CovarianceMatrix

class PCANode(SignalNode):
    """PCANode receives an input signal and filters it through
    the most significatives of its principal components.
    More information about Principal Component Analysis, a.k.a. discrete
    Karhunen-Loeve transform can be found among others in
    I.T. Jolliffe, Principal Component Analysis, Springer-Verlag (1986)."""
    
    def __init__(self, input_dim = None, output_dim = None, typecode = None):
        """The number of principal components to be kept can be specified as
        'output_dim' directly (e.g. 'output_dim=10' means 10 components
        are kept) or by the fraction of variance to be explained
        (e.g. 'output_dim=0.95' means that as many components as necessary
        will be kept in order to explain 95% of the input variance)."""
        
        super(PCANode, self).__init__(input_dim, output_dim, typecode)

        # empirical covariance matrix, updated during the training phase
        self._cov_mtx = CovarianceMatrix(typecode)
        
    def _check_output(self, y):
        # check output rank
        if not numx.rank(y) == 2:
            error_str = "y has rank %d, should be 2"\
                        %(numx.rank(y))
            raise SignalNodeException, error_str

        if y.shape[1]==0 or y.shape[1]>self._output_dim:
            error_str = "y has dimension %d, should be 0<y<=%d" \
                        % (y.shape[1], self._output_dim)
            raise SignalNodeException, error_str

    def set_output_dim(self, output_dim):
        """Set output dimensions.
        This only works _before_ the end of the training phase."""
        if self.is_training():
            self._set_default_outputdim(output_dim)
        else:
            errstr = "The output dimension cannot be changed "+ \
                     "after the training phase."
            raise TrainingFinishedException, errstr

    def get_supported_typecodes(self):
        return ['f','d']
    
    def get_explained_variance(self):
        """Return the fraction of the original variance that can be
        explained by self._output_dim. PCA components.
        If for example output_dim has been set to 0.95, the explained
        variance could be something like 0.958..."""
        return self.explained_variance
    
    def train(self, x):
        super(PCANode, self).train(x)
        # update the covariance matrix
        x = self._refcast(x)
        self._cov_mtx.update(x)
        
    def stop_training(self):
        super(PCANode, self).stop_training()

        ##### request the covariance matrix and clean up
        cov_mtx, avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx
        
        # this is a bit counterintuitive, as it reshapes the average vector to
        # be a matrix. in this way, however, we spare the reshape
        # operation every time that 'execute' is called.
        self.avg = numx.reshape(avg, (1, avg.shape[0]))

        ##### compute the principal components
        # if the number of principal components to keep is not specified,
        # keep all components
        if not self._output_dim:
            self._output_dim = self._input_dim

        ## define the range of eigenvalues to compute
        # if the number of principal components to keep has been
        # specified directly
        if self._output_dim>=1:
            # (eigenvalues sorted in ascending order)
            rng = (self._input_dim-self._output_dim+1, self._input_dim)
        # otherwise, the number of principal components to keep has been
        # specified by the fraction of variance to be explained
        else:
            rng = None

        ## compute and sort the eigenvalues
        # compute the eigenvectors of the covariance matrix (inplace)
        # (eigenvalues sorted in ascending order)
        try:
            d, v = symeig(cov_mtx, range = rng, overwrite = 1)
        except LeadingMinorException, exception:
            errstr = str(exception)+"\n Covariance matrix may be singular."
            raise SignalNodeException,errstr
        
        # sort by descending order
        d = numx.take(d, range(d.shape[0]-1,-1,-1))
        v = v[:,::-1]

        ## compute the explained variance
        # if the number of principal components to keep has been
        # specified directly
        if self._output_dim>=1:
            # there is no way to tell what the explained variance is, since we
            # didn't compute all eigenvalues
            self.explained_variance = None
        # otherwise, the number of principal components to keep has
        # been specified by the fraction of variance to be explained
        else:
            # total variance
            vartot = numx.sum(d)
            # cumulative variance (percent)
            varcum = numx.cumsum(d/vartot, axis = 0)
            # select only the relevant eigenvalues
            # number of relevant eigenvalues
            neigval = numx.searchsorted(varcum, self._output_dim)+1
            self.explained_variance = varcum[neigval]
            # cut
            d = d[0:neigval]
            v = v[:,0:neigval]
            # define the new output dimension
            self._output_dim = neigval
            
        ## store the eigenvalues
        self.d = d

        ## store the eigenvectors
        self.v = v

    def get_projmatrix(self ,transposed=1):
        """Return the projection matrix."""
        self._check_if_training()
        if transposed:
            return self.v
        return numx.transpose(self.v)

    def get_recmatrix(self, transposed=1):
        """Return the back-projection matrix (i.e. the reconstruction matrix).
        """
        self._check_if_training()
        if transposed:
            return numx.transpose(self.v)
        return self.v

    def execute(self, x, n = None):
        """Project the input on the first 'n' principal components.
        If 'n' is not set, use all available components."""
        super(PCANode, self).execute(x)
        x = self._refcast(x)
        if n is not None:
            return mult(x-self.avg, self.v[:,:n])
        return mult(x-self.avg, self.v)

    def inverse(self, y, n = None):
        """Project 'y' to the input space using the first 'n' components.
        If 'n' is not set, use all available components."""
        super(PCANode, self).inverse(y)
        y = self._refcast(y)
        if n is None:
            n = y.shape[1]
        if n>self._output_dim:
            error_str = "y has dimension %d, should be at most %d" \
                        %(n, self._output_dim)
            raise SignalNodeException, error_str
        
        v = self.get_recmatrix()
        return mult(y, v[:n,:])+self.avg


class WhiteningNode(PCANode):
    """WhiteningNode receives an input signal and 'whiten' it by filtering
    it through the most significatives of its principal components. All output
    signals have zero mean, unit variance and are decorrelated."""
    
    def stop_training(self):
        super(WhiteningNode, self).stop_training()

        ##### whiten the filters
        # self.v is now the _whitening_ matrix
        self.v = self.v / numx.sqrt(self.d)

    def get_eigenvectors(self):
        """Return the eigenvectors of the covariance matrix."""
        self._check_if_training()
        return numx.sqrt(self.d)*self.v

    def get_recmatrix(self, transposed=1):
        """Return the back-projection matrix (i.e. the reconstruction matrix).
        """
        self._check_if_training()
        v_inverse = self.v*self.d
        if transposed:
            return numx.transpose(v_inverse)
        return v_inverse
        
