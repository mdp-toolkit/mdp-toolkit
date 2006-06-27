import mdp
from mdp import numx, numx_linalg, utils
from mdp.utils import mult, CovarianceMatrix
import warnings

take, put, diag, ravel = numx.take, numx.put, numx.diag, numx.ravel
sqrt, tr, inv, det = numx.sqrt, numx.transpose, utils.inv, numx_linalg.det
normal = mdp.numx_rand.normal

# decreasing likelihood message
_LHOOD_WARNING = 'Likelihood decreased in FANode. This is probably due '+\
                 'to some numerical errors.'
warnings.filterwarnings('always', _LHOOD_WARNING, mdp.MDPWarning)


class FANode(mdp.Node):

    def __init__(self, tol=1e-4, max_cycles=100, verbose=False,
                 input_dim=None, output_dim=None, dtype=None):
        """Perform Factor Analysis.

        The current implementation should be most efficient for long
        data sets with

        tol -- tolerance (minimum change in log-likelihood)
        max_cycles -- maximum number of EM cycles
        verbose -- if True, print log-likelihood during the EM-cycles

        More information about Factor Analysis can be found in
        Max Welling's classnotes:
        http://www.ics.uci.edu/~welling/classnotes/classnotes.html ,
        under the chapter 'Linear Models'.
        """
        # Notation as in Max Welling's notes
        super(FANode, self).__init__(input_dim, output_dim, dtype)
        self.tol = tol
        self.max_cycles = max_cycles
        self.verbose = verbose
        self._cov_mtx = CovarianceMatrix(dtype, bias=True)
    
    def _train(self, x):
        # update the covariance matrix
        self._cov_mtx.update(x)
        
    def _stop_training(self):
        #### some definitions
        verbose = self.verbose
        typ = self.dtype
        tol = self.tol
        d = self.input_dim
        # if the number of latent variables is not specified,
        # set it equal to the number of input components
        if not self.output_dim: self.output_dim = d
        k = self.output_dim
        # indices of the diagonal elements of a dxd or kxk matrix
        idx_diag_d = [i*(d+1) for i in range(d)]
        idx_diag_k = [i*(k+1) for i in range(k)]
        # constant term in front of the log-likelihood
        const = -d/2. * numx.log(2.*numx.pi)

        ##### request the covariance matrix and clean up
        cov_mtx, mu, tlen = self._cov_mtx.fix()
        del self._cov_mtx
        # 'bias' the covariance matrix
        # (i.e., cov_mtx = 1/tlen sum(x_t x_t^T) instead of 1/(tlen-1))
        cov_diag = diag(cov_mtx)

        ##### initialize the parameters
        # noise variances
        sigma = cov_diag
        # loading factors
        # Zoubin uses the determinant of cov_mtx^1/d as scale but it's
        # too slow for large matrices. Ss the product of the diagonal a good
        # approximation?
        #scale = det(cov_mtx)**(1./d)
        scale = numx.product(sigma)**(1./d)
        A = normal(0., sqrt(scale/k), size=(d,k)).astype(typ)

        ##### EM-cycle
        lhood_curve = []
        base_lhood = None
        old_lhood = -numx.inf
        for t in xrange(self.max_cycles):
            ## compute B = (A A^T + Sigma)^-1
            ##???###
            # copy to make it contiguous, otherwise .ravel() does not work
            B = mult(A, tr(A)).copy()
            # B += diag(sigma), avoid computing diag(sigma) which is dxd
            put(B.ravel(), idx_diag_d, take(B.ravel(), idx_diag_d)+sigma)
            # this quantity is used later for the log-likelihood
            # abs is there to avoid numerical errors when det < 0 
            log_det_B = numx.log(abs(det(B)))
            # end the computation of B
            B = inv(B)
           
            ## other useful quantities
            trA_B = mult(tr(A), B)
            trA_B_cov_mtx = mult(trA_B, cov_mtx)
            
            ##### E-step
            ## E_yyT = E(y_n y_n^T | x_n)
            E_yyT = - mult(trA_B, A) + mult(trA_B_cov_mtx, tr(trA_B))
            # E_yyT += numx.eye(k)
            put(E_yyT.ravel(), idx_diag_k, take(E_yyT.ravel(), idx_diag_k)+1.)
            
            ##### M-step
            A = mult(tr(trA_B_cov_mtx), inv(E_yyT))
            sigma = cov_diag - diag(mult(A, trA_B_cov_mtx))

            ##### log-likelihood
            trace_B_cov = numx.sum(ravel(B*tr(cov_mtx)))
            # this is actually likelihood/tlen.
            # cast to float explicitly. Numarray doesn't like
            # 0-D arrays to be used in logical expressions
            lhood = float(const - 0.5*log_det_B - 0.5*trace_B_cov)
            if verbose: print 'cycle',t,'log-lhood:',lhood

            ##### convergence criterion
            if base_lhood is None: base_lhood = lhood
            else:
                # convergence criterion
                if (lhood-base_lhood)<(1.+tol)*(old_lhood-base_lhood): break
                if lhood < old_lhood:
                    # this should never happen
                    # it sometimes does, e.g. if the noise is extremely low,
                    # because of numerical problems
                    warnings.warn(_LHOOD_WARNING, mdp.MDPWarning)
            old_lhood = lhood
            lhood_curve.append(lhood)

        self.tlen = tlen
        self.A = A
        self.mu = numx.reshape(mu, (1, d))
        self.sigma = sigma
        
        ## MAP matrix
        # compute B = (A A^T + Sigma)^-1
        B = mult(A, tr(A)).copy() 
        put(B.ravel(), idx_diag_d, take(B.ravel(), idx_diag_d)+sigma)
        B = inv(B)
        self.E_y_mtx = mult(tr(B), A)
        
        self.lhood = lhood_curve

    def _execute(self, x):
        """Return the Maximum A-Posteriori estimate for the latent
        variables."""
        return mult(x-self.mu, self.E_y_mtx)

    def _inverse(self, y, noise = False):
        """Generate observations x.

        noise -- if True, generation includes the estimated noise."""
        res = mult(y, tr(self.A))+self.mu
        if noise:
            ns = numx_rand.normal(0., self.sigma,
                              size=(y.shape[0], self.input_dim))
            res += self.refcast(ns)
        return res

