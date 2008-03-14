from mdp import numx, Node, NodeException, Cumulator
from mdp.utils import mult
from mdp.nodes import PCANode
sqrt = numx.sqrt

class NIPALSNode(PCANode, Cumulator):
    """
    Perform Principal Component Analysis using the NIPALS algorithm. 

    Internal variables of interest:
    self.avg -- Mean of the input data (available after training)
    self.v -- Transposed of the projection matrix (available after training)    

    Reference for NIPALS (Nonlinear Iterative Partial Least Squares):
    Wold, H.
    Nonlinear estimation by iterative least squares procedures
    in David, F. (Editor), Research Papers in Statistics, Wiley,
    New York, pp 411-444 (1966).
    
    More information about Principal Component Analysis, a.k.a. discrete
    Karhunen-Loeve transform can be found among others in
    I.T. Jolliffe, Principal Component Analysis, Springer-Verlag (1986).

    Code contributed by:
    Michael Schmuker, Susanne Lezius, and Farzad Farkhooi (2008).
    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 conv = 1e-8, max_it = 100000):
        """
        Unlike the standard PCANode, the number of principal components can only
        be specified as an integer number.

        Other Arguments:
           conv   - convergence threshold for the residual error.
           max_it - maximum number of iterations
           
        """
        if output_dim <= 1 and isinstance(output_dim, float): 
            raise NotImplementedError
        else: 
            self.desired_variance = None 

        super(NIPALSNode, self).__init__(input_dim, output_dim, dtype)
        self.conv = conv
        self.max_it = max_it
        self.explained_variance = None
        self.d = None

    def _train(self, x):
        Cumulator._train(self, x)
        
    def _stop_training(self, debug=False):
        # debug argument is ignored but needed by the base class
        Cumulator._stop_training(self)
        if self.output_dim is None:
            self.output_dim = self.input_dim

        X = self.data
        conv = self.conv
        dtype = self.dtype
        mean = X.mean(axis=0)
        self.avg = mean
        max_it = self.max_it

        # remove mean and variance
        X -= mean
        #X /= std

        eigenv = numx.zeros((self.output_dim,self.input_dim), dtype=dtype)
        for i in range(self.output_dim):
            it = 0
            # first score vector t is initialized to first column in X
            t = X[:,0]
            # initialize difference
            diff = conv + 1
            while diff > conv:
                # increase iteration counter
                it += 1 
                # Project X onto t to find corresponding loading p
                # and normalize loading vector p to length 1
                p = mult(X.T,t)/mult(t,t)
                p /= sqrt(mult(p,p))

                # project X onto p to find corresponding score vector t_new
                t_new = mult(X, p)
                # difference between new and old score vector
                tdiff = t_new - t
                diff = (tdiff*tdiff).sum()
                t = t_new
                if it > max_it:
                    msg = 'PC#%d No convergence after %d iterations.'%(i,max_it)
                    raise NodeException(msg)

            # store ith eigenvector in result matrix
            eigenv[i, :] = p
            # remove the estimated principal component from X
            X -= numx.outer(t,p)

        self.v = eigenv.T
