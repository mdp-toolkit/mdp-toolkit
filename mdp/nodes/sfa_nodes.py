import mdp
from mdp import numx, utils, Node, \
     NodeException, TrainingFinishedException
from mdp.utils import mult, pinv, symeig, CovarianceMatrix, QuadraticForm, \
                      SymeigException

class SFANode(Node):
    """Extract the slowly varying components from the input data.
    More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
    Learning of Invariances, Neural Computation, 14(4):715-770 (2002).

    Internal variables of interest:
    self.avg -- Mean of the input data (available after training)
    self.sf -- Matrix of the SFA filters (available after training)
    self.d -- Delta values corresponding to the SFA components
              (generalized eigenvalues).
              (See the docs of the 'get_eta_values' method for
              more information)
    """
    
    def __init__(self, input_dim=None, output_dim=None, dtype=None):
        super(SFANode, self).__init__(input_dim, output_dim, dtype)

        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(dtype)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(dtype)

        # set routine for eigenproblem
        self._symeig = symeig

    
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative."""
        # this is faster than a linear_filter or a weave-inline solution
        return x[1:,:]-x[:-1,:]
        
    def _train(self, x):
        ## update the covariance matrices
        # cut the final point to avoid a trivial solution in special cases
        self._cov_mtx.update(x[:-1,:])
        self._dcov_mtx.update(self.time_derivative(x))

    def _stop_training(self, debug=False):
        ##### request the covariance matrices and clean up
        self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx
        self.dcov_mtx, davg, dtlen = self._dcov_mtx.fix()
        del self._dcov_mtx

        if self.output_dim is not None and self.output_dim <= self.input_dim:
            # (eigenvalues sorted in ascending order)
            rng = (1, self.output_dim)
        else:
            # otherwise, keep all output components
            rng = None        
        
        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        try:
            self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx,
                                     range=rng, overwrite=(not debug))
        except SymeigException, exception:
            errstr = str(exception)+"\n Covariance matrices may be singular."
            raise NodeException, errstr

        # delete covariance matrix if no exception occurred
        del self.cov_mtx
        del self.dcov_mtx
        
    def _execute(self, x, range=None):
        if range:
            if isinstance(range, (list, tuple)):
                sf = self.sf[:,range[0]:range[1]]
            else:
                sf = self.sf[:,0:range]
        else:
            sf = self.sf
        return mult(x-self.avg, sf)

    def execute(self, x, range=None):
        """Compute the output of the slowest functions.
        if 'range' is a number, then use the first 'range' functions.
        if 'range' is the interval=(i,j), then use all functions
                   between i and j."""
        return super(SFANode, self).execute(x, range)

    def _inverse(self, y):
        return mult(y, pinv(self.sf))+self.avg

    def get_eta_values(self, t=1):
        """Return the eta values of the slow components learned during
        the training phase. If the training phase has not been completed
        yet, call stop_training.
        
        The delta value of a signal is a measure of its temporal
        variation, and is defined as the mean of the derivative squared,
        i.e. delta(x) = mean(dx/dt(t)^2).  delta(x) is zero if
        x is a constant signal, and increases if the temporal variation
        of the signal is bigger.
        
        The eta value is a more intuitive measure of temporal variation,
        defined as
            eta(x) = t/(2*pi) * sqrt(delta(x))
        If x is a signal of length 't' which consists of a sine function
        that accomplishes exactly N oscillations, then eta(x)=N.
        
        Input arguments:
        t -- Time units (e.g., t=0.01 if you sample at 100Hz)
        """
        if self.is_training(): self.stop_training()
        return self._refcast(t/(2*numx.pi)*numx.sqrt(self.d))

class SFA2Node(SFANode):
    """Get an input signal, expand it in the space of
    inhomogeneous polynomials of degree 2 and extract its slowly varying
    components. The 'get_quadratic_form' method returns the input-output
    function of one of the learned unit as a QuadraticForm object.
    See the documentation of mdp.utils.QuadraticForm for additional
    information.

    More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
    Learning of Invariances, Neural Computation, 14(4):715-770 (2002)."""

    def __init__(self, input_dim=None, output_dim=None, dtype=None):
        self._expnode = mdp.nodes.QuadraticExpansionNode(input_dim=input_dim,
                                                         dtype=dtype)
        super(SFA2Node, self).__init__(input_dim, output_dim, dtype)

    def is_invertible(self):
        """Return True if the node can be inverted, False otherwise."""
        return False

    def _set_input_dim(self, n):
        self._expnode.input_dim = n
        self._input_dim = n
        
    def _train(self, x):
        # expand in the space of polynomials of degree 2
        super(SFA2Node, self)._train(self._expnode(x))

    def _stop_training(self, debug=False):
        super(SFA2Node, self)._stop_training(debug)

        # set the output dimension if necessary
        if self.output_dim is None:
            self.output_dim = self._expnode.output_dim

    def _execute(self, x, range=None):
        """Compute the output of the slowest functions.
        if 'range' is a number, then use the first 'range' functions.
        if 'range' is the interval=(i,j), then use all functions
                   between i and j."""
        return super(SFA2Node, self)._execute(self._expnode(x))

    def get_quadratic_form(self, nr):
        """
        Return the matrix H, the vector f and the constant c of the
        quadratic form 1/2 x'Hx + f'x + c that defines the output
        of the component 'nr' of the SFA node.
        """

        self._if_training_stop_training()

        sf = self.sf[:, nr]
        c = -mdp.utils.mult(self.avg, sf)
        N = self.input_dim
        f = sf[:N]
        H = numx.zeros((N,N), dtype = self.dtype)
        k = N
        for i in range(N):
            for j in range(N):
                if j > i:
                    H[i,j] = sf[k]
                    k = k+1
                elif j == i:
                    H[i,j] = 2*sf[k]
                    k = k+1
                else:
                    H[i,j] = H[j,i]

        return QuadraticForm(H, f, c, dtype = self.dtype)

               

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
