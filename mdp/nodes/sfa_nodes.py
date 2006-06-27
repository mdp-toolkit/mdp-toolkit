from mdp import numx, utils, Node, \
     NodeException, TrainingFinishedException
from mdp.utils import mult, pinv, symeig, CovarianceMatrix
                      #, LeadingMinorException

class SFANode(Node):
    """SFANode receives an input signal and extracts its slowly varying
    components. More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
    Learning of Invariances, Neural Computation, 14(4):715-770 (2002)."""
    
    def __init__(self, input_dim=None, output_dim=None, dtype=None):
        super(SFANode, self).__init__(input_dim, output_dim, dtype)

        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(dtype)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(dtype)

    def _get_supported_dtypes(self):
        return ['f','d']

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative."""
        # this is faster than a linear_filter or a weave-inline solution
        return x[1:,:]-x[:-1,:]
        
    def _train(self, x):
        ## update the covariance matrices
        # cut the final point to avoid a trivial solution in special cases
        self._cov_mtx.update(x[:-1,:])
        self._dcov_mtx.update(self.time_derivative(x))

    def _stop_training(self):
        ##### request the covariance matrices and clean up
        cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx
        dcov_mtx, davg, dtlen = self._dcov_mtx.fix()
        del self._dcov_mtx

        # if the number of output components to keep is not specified,
        # keep all components
        if not self.output_dim:
            self.output_dim = self.input_dim

        if self.output_dim < self.input_dim:
            # (eigenvalues sorted in ascending order)
            rng = (1, self.output_dim)
        else:
            # otherwise, keep all output components
            rng = None
        
        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        try:
            self.d, self.sf = symeig(dcov_mtx, cov_mtx, range=rng, overwrite=1)
        #except LeadingMinorException, exception:
        except exception:
            errstr = str(exception)+"\n Covariance matrices may be singular."
            raise NodeException,errstr

        # check that we didn't get negative eigenvalues,
        # if this is the case the covariance matrix may be singular
        if min(self.d) <= 0:
            errs="Got negative eigenvalues: Covariance matrix may be singular."
            raise NodeException, errs 
        
    def _execute(self, x, range=None):
        """Compute the output of the slowest functions.
        if 'range' is a number, then use the first 'range' functions.
        if 'range' is the interval=(i,j), then use all functions
                   between i and j."""
        if range:
            if isinstance(range, (list, tuple)):
                sf = self.sf[:,range[0]:range[1]]
            else:
                sf = self.sf[:,0:range]
        else:
            sf = self.sf
        return mult(x-self.avg, sf)

    def _inverse(self, y):
        return mult(y, pinv(self.sf))+self.avg

### old weave inline code to perform the time derivative

# weave C code executed in the function SfaNode.time_derivative
## _TDERIVATIVE_1ORDER_CCODE = """
##   for( int i=0; i<columns; i++ ) {
##     for( int j=0; j<rows-1; j++ ) {
##       deriv(j,i) = x(j+1,i)-x(j,i);
##     }
##   }
## """

# it was called like that:
## def time_derivative(self, x):
##     rows = x.shape[0]
##     columns = x.shape[1]
##     deriv = numx.zeros((rows-1, columns), dtype=self.dtype)

##     weave.inline(_TDERIVATIVE_1ORDER_CCODE,['rows','columns','deriv','x'],
##                  type_factories = weave.blitz_tools.blitz_type_factories,
##                  compiler='gcc',extra_compile_args=['-O3']);
##     return deriv
