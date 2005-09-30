from lcov import CovarianceMatrix
from mdp import numx, utils, SignalNode, \
     SignalNodeException, TrainingFinishedException
from mdp.utils import mult, pinv, symeig, LeadingMinorException

class SFANode(SignalNode):
    """SFANode receives an input signal and extracts its slowly varying
    components. More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
    Learning of Invariances, Neural Computation, 14(4):715-770 (2002)."""
    
    def __init__(self, input_dim=None, output_dim=None, typecode=None):
        super(SFANode, self).__init__(input_dim, output_dim, typecode)

        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(typecode)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(typecode)

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

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative."""
        # this is faster than a linear_filter or a weave-inline solution
        return x[1:,:]-x[:-1,:]
        
    def train(self, x):
        super(SFANode, self).train(x)
        x = self._refcast(x)
        ## update the covariance matrices
        # cut the final point to avoid a trivial solution in special cases
        self._cov_mtx.update(x[:-1,:])
        self._dcov_mtx.update(self.time_derivative(x))

    def stop_training(self):
        super(SFANode, self).stop_training()

        ##### request the covariance matrices and clean up
        cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx
        dcov_mtx, davg, dtlen = self._dcov_mtx.fix()
        del self._dcov_mtx

        # if the number of output components to keep is not specified,
        # keep all components
        if not self._output_dim:
            self._output_dim = self._input_dim

        if self._output_dim < self._input_dim:
            # (eigenvalues sorted in ascending order)
            rng = (1, self._output_dim)
        else:
            # otherwise, keep all output components
            rng = None
        
        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        try:
            self.d, self.sf = symeig(dcov_mtx, cov_mtx, range=rng, overwrite = 1)
        except LeadingMinorException, exception:
            errstr = str(exception)+"\n Covariance matrices may be singular."
            raise SignalNodeException,errstr
            
    def execute(self, x, range=None):
        """Compute the output of the slowest functions.
        if 'range' is a number, then use the first 'range' functions.
        if 'range' is the interval=(i,j), then use all functions
                   between i and j."""
        super(SFANode, self).execute(x)
        x = self._refcast(x)
        
        if range:
            if isinstance(range, (list, tuple)):
                sf = self.sf[:,range[0]:range[1]]
            else:
                sf = self.sf[:,0:range]
        else:
            sf = self.sf
        return mult(x-self.avg, sf)

    def inverse(self, y):
        SignalNode.inverse(self, y)
        y = self._refcast(y)
        return mult(y, pinv(self.sf))+self.avg

# backwards compatibility
SfaNode = SFANode


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
##     deriv = numx.zeros((rows-1, columns), typecode=self._typecode)

##     weave.inline(_TDERIVATIVE_1ORDER_CCODE,['rows','columns','deriv','x'],
##                  type_factories = weave.blitz_tools.blitz_type_factories,
##                  compiler='gcc',extra_compile_args=['-O3']);
##     return deriv
