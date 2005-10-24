import mdp
from mdp import numx, numx_linalg
from mdp.utils import mult, normal
from lcov import CovarianceMatrix
import warnings

take, put, diag, ravel = numx.take, numx.put, numx.diag, numx.ravel
sqrt, tr, inv, det = numx.sqrt, numx.transpose,numx_linalg.inv,numx_linalg.det

# decreasing likelihood message
_LHOOD_WARNING = 'Likelihood decreased in FANode. This is probably due '+\
                'to some numerical errors.'
warnings.filterwarnings('always', _LHOOD_WARNING, mdp.MDPWarning)

class FANode(mdp.FiniteNode):

    def __init__(self, tol=1e-4, max_cycles=100, verbose=False,
                 input_dim=None, output_dim=None, typecode=None):
        """Perform Factor Analysis.

        tol -- tolerance (change in log-likelihood)
        max_cycles -- maximum number of EM cycles
        verbose -- if True, print log-likelihood during the EM-cycles
        """
        # Notation as in Max Welling's notes
        super(FANode, self).__init__(input_dim, output_dim, typecode)
        self.tol = tol
        self.max_cycles = max_cycles
        self.verbose = verbose
        self._cov_mtx = CovarianceMatrix(typecode)
    
    def _train(self, x):
        # update the covariance matrix
        self._cov_mtx.update(x)
        
    def _stop_training(self):
        #### some definitions
        verbose = self.verbose
        type = self._typecode
        one = self._scast(1.)
        tol = self.tol
        d = self._input_dim
        # if the number of latent variables is not specified,
        # set it equal to the number of input components
        if not self._output_dim: self._output_dim = d
        k = self._output_dim
        # indices of the diagonal elements of a dxd or kxk matrix
        idx_diag_d = [i*(d+1) for i in range(d)]
        idx_diag_k = [i*(k+1) for i in range(k)]
        # constant term in front of the log-likelihood
        const= self._scast(-d/2. * numx.log(2.*numx.pi))

        ##### request the covariance matrix and clean up
        cov_mtx, mu, tlen = self._cov_mtx.fix()
        del self._cov_mtx
        # 'bias' the covariance matrix
        cov_mtx *= (tlen-self._scast(1.))/tlen
        cov_diag = diag(cov_mtx).astype(type)

        ##### initialize the parameters
        # noise variances
        sigma = cov_diag
        # loading factors
        # Zoubin uses the determinant of cov_mtx^1/d as scale but it's
        # too slow. is the product of the diagonal a good approximation?
        #scale = det(cov_mtx)**(1./d)
        scale = numx.product(sigma)**(1./d)
        A = normal(0., sqrt(scale/k), shape=(d,k)).astype(type)

        ##### EM-cycle
        lhood_curve = []
        base_lhood = None
        old_lhood = -numx.inf
        for t in range(self.max_cycles):
            ## compute B = (A A^T + Sigma)^-1
            # copy to make it contiguous, otherwise .flat does not work
            B = mult(A, tr(A)).copy()
            # B += diag(sigma), avoid computing diag(sigma) which is dxd
            put(B.flat, idx_diag_d, take(B.flat, idx_diag_d)+sigma)
            # this quantity is used later for the log-likelihood
            log_det_inv_B = numx.log(det(B))
            B = inv(B)
            
            ## other useful quantities
            trA_B = mult(tr(A), B)
            trA_B_cov_mtx = mult(trA_B, cov_mtx)
            
            ##### E-step
            ## E_yyT = E(y_n y_n^T | x_n)
            E_yyT = - mult(trA_B, A) + mult(trA_B_cov_mtx, tr(trA_B))
            # E_yyT += numx.eye(k)
            put(E_yyT.flat, idx_diag_k, take(E_yyT.flat, idx_diag_k)+one)
            
            ##### M-step
            A = mult(tr(trA_B_cov_mtx), inv(E_yyT))
            sigma = cov_diag - diag(mult(A, trA_B_cov_mtx))

            ##### log-likelihood
            #trace_B_cov = numx.sum(ravel(tr(B*cov_mtx)))
            trace_B_cov = numx.sum(ravel(B*tr(cov_mtx)))
            # this is actually likelihood over tlen
            lhood = const - 0.5*log_det_inv_B - 0.5*trace_B_cov
            if verbose: print 'cycle',t,'log-lhood:',lhood

            ##### convergence criterion
            if base_lhood is None: base_lhood = lhood
            else:
                # convergence criterion
                if (lhood-base_lhood)<(1.+tol)*(old_lhood-base_lhood): break
                if lhood<old_lhood:
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
        put(B.flat, idx_diag_d, take(B.flat, idx_diag_d)+sigma)
        B = inv(B)
        self.E_y_mtx = mult(tr(B), A)
        
        self.lhood = lhood_curve

    def _execute(self, x):
        """Return the Maximum A-Posteriori estimate for the latent
        variables."""
        return mult(x-self.mu, self.E_y_mtx)

    def _inverse(self, y, noise=False):
        """Generate observation x.

        noise -- if True, generation includes noise."""
        res = mult(y, tr(self.A))+self.mu
        if noise:
            res += mdp.utils.normal(0., self.sigma,
                                    shape=(y.shape[0], self._input_dim))
        return res
