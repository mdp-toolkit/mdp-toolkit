import mdp
numx = mdp.numx

def nmonomials(degree, nvariables):
    """Return the number of monomials of a given degree in a given number
    of variables."""
    return int(mdp.utils.comb(nvariables+degree-1, degree))

def expanded_dim(degree, nvariables):
    """Return the size of a vector of dimension 'nvariables' after
    a polynomial expansion of degree 'degree'."""
    return int(mdp.utils.comb(nvariables+degree, degree))-1

class _ExpansionNode(mdp.Node):
    
    def __init__(self, input_dim = None, dtype = None):
        super(_ExpansionNode, self).__init__(input_dim, None, dtype)

    def expanded_dim(self, dim):
        return dim
        
    def is_trainable(self):
        return False

    def is_invertible(self):
        return False
            
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = self.expanded_dim(n)

    def _set_output_dim(self, n):
        msg = "Output dim cannot be set explicitly!"
        raise mdp.NodeException, msg

class PolynomialExpansionNode(_ExpansionNode):
    """Perform expansion in a polynomial space."""

    def __init__(self, degree, input_dim = None, dtype = None):
        """
        Input arguments:
        degree -- degree of the polynomial space where the input is expanded
        """
        self._degree = int(degree)
        super(PolynomialExpansionNode, self).__init__(input_dim, dtype)

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return mdp.utils.get_dtypes('AllFloat') + \
               mdp.utils.get_dtypes('AllInteger')
    
    def expanded_dim(self, dim):
        """Return the size of a vector of dimension 'dim' after
        a polynomial expansion of degree 'self._degree'."""
        return expanded_dim(self._degree, dim)
    
    def _execute(self, x):
        degree = self._degree
        dim = self.input_dim
        n = x.shape[1]
        
        # preallocate memory
        dexp = numx.zeros((self.output_dim, x.shape[0]), \
                           dtype=self.dtype)
        # copy monomials of degree 1
        dexp[0:n,:] = x.T

        k = n
        prec_end = 0
        next_lens = numx.ones((dim+1,))
        next_lens[0] = 0
        for i in range(2,degree+1):
            prec_start = prec_end
            prec_end += nmonomials(i-1, dim)
            prec = dexp[prec_start:prec_end,:]

            lens = next_lens[:-1].cumsum(axis=0)
            next_lens = numx.zeros((dim+1,))
            for j in range(dim):
                factor = prec[lens[j]:,:]
                len = factor.shape[0]
                dexp[k:k+len,:] = x[:,j] * factor
                next_lens[j+1] = len
                k = k+len

        return dexp.T
        
class QuadraticExpansionNode(PolynomialExpansionNode):
    """Perform expansion in the space formed by all linear and quadratic
    monomials.
    QuadraticExpansionNode() is equivalent to a PolynomialExpansionNode(2)"""

    def __init__(self, input_dim = None, dtype = None):
        super(QuadraticExpansionNode, self).__init__(2, input_dim = input_dim,
                                                     dtype = dtype)
    

### old weave inline code to perform a quadratic expansion

# weave C code executed in the function QuadraticExpansionNode.execute
## _EXPANSION_POL2_CCODE = """
##   // first of all, copy the linear part
##   for( int i=0; i<columns; i++ ) {
##     for( int l=0; l<rows; l++ ) {
##       dexp(l,i) = x(l,i);
##     }
##   }

##   // then, compute all monomials of second degree
##   int k=columns;
##   for( int i=0; i<columns; i++ ) {
##     for( int j=i; j<columns; j++ ) {
##       for( int l=0; l<rows; l++ ) {
##         dexp(l,k) = x(l,i)*x(l,j);
##       }
##       k++;
##     }
##   }
## """

# it was called like that:
##     def execute(self, x):
##         mdp.Node.execute(self, x)

##         rows = x.shape[0]
##         columns = self.input_dim
##         # dexp is going to contain the expanded signal
##         dexp = numx.zeros((rows, self.output_dim), dtype=self._dtype)

##         # execute the inline C code
##         weave.inline(_EXPANSION_POL2_CCODE,['rows','columns','dexp','x'],
##                  type_factories = weave.blitz_tools.blitz_type_factories,
##                  compiler='gcc',extra_compile_args=['-O3']);
        
##         return dexp
        
