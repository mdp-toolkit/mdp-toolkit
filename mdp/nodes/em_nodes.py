from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
__docformat__ = "restructuredtext en"

import mdp
from mdp import numx, numx_linalg, utils, NodeException
from mdp.utils import mult, CovarianceMatrix
import warnings

sqrt, inv, det = numx.sqrt, utils.inv, numx_linalg.det
normal = mdp.numx_rand.normal

# decreasing likelihood message
_LHOOD_WARNING = ('Likelihood decreased in FANode. This is probably due '
                  'to some numerical errors.')
warnings.filterwarnings('always', _LHOOD_WARNING, mdp.MDPWarning)

class FANode(mdp.Node):
    """Perform Factor Analysis.

    The current implementation should be most efficient for long
    data sets: the sufficient statistics are collected in the
    training phase, and all EM-cycles are performed at
    its end.

    The ``execute`` method returns the Maximum A Posteriori estimate
    of the latent variables. The ``generate_input`` method generates
    observations from the prior distribution.

    :ivar mu: Mean of the input data (available after training).

    :ivar A: Generating weights (available after training).

    :ivar E_y_mtx: Weights for Maximum A Posteriori inference.

    :ivar sigma: Vector of estimated variance of the noise
        for all input components.
    
    |
    
    .. admonition:: Reference
    
        More information about Factor Analysis can be found in
        Max Welling's classnotes:
        http://www.ics.uci.edu/~welling/classnotes/classnotes.html ,
        in the chapter 'Linear Models'.
    """
    def __init__(self, tol=1e-4, max_cycles=100, verbose=False,
                 input_dim=None, output_dim=None, dtype=None):
        """Initializes an object of type 'FANode'.
        
        :param tol: Tolerance (minimum change in log-likelihood before exiting
            the EM algorithm).
        :type tol: float
        
        :param max_cycles: Maximum number of EM cycles/
        :type max_cycles: int
        
        :param verbose: If true, print log-likelihood during the EM-cycles.
        :type verbose: bool
        
        :param input_dim: The input dimensionality.
        :type input_dim: int
        
        :param output_dim: The output dimensionality.
        :type output_dim: int
        
        :param dtype: The datatype.
        :type dtype: numpy.dtype or str
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
        if not self.output_dim:
            self.output_dim = d
        k = self.output_dim
        # indices of the diagonal elements of a dxd or kxk matrix
        idx_diag_d = [i*(d+1) for i in range(d)]
        idx_diag_k = [i*(k+1) for i in range(k)]
        # constant term in front of the log-likelihood
        const = -d/2. * numx.log(2.*numx.pi)

        ##### request the covariance matrix and clean up
        cov_mtx, mu, tlen = self._cov_mtx.fix()
        del self._cov_mtx
        cov_diag = cov_mtx.diagonal()

        ##### initialize the parameters
        # noise variances
        sigma = cov_diag
        # loading factors
        # Zoubin uses the determinant of cov_mtx^1/d as scale but it's
        # too slow for large matrices. Is the product of the diagonal a good
        # approximation?
        if d<=300:
            scale = det(cov_mtx)**(old_div(1.,d))
        else:
            scale = numx.product(sigma)**(old_div(1.,d))
        if scale <= 0.:
            err = ("The covariance matrix of the data is singular. "
                   "Redundant dimensions need to be removed.")
            raise NodeException(err)

        A = normal(0., sqrt(old_div(scale,k)), size=(d, k)).astype(typ)

        ##### EM-cycle
        lhood_curve = []
        base_lhood = None
        old_lhood = -numx.inf
        for t in range(self.max_cycles):
            ## compute B = (A A^T + Sigma)^-1
            B = mult(A, A.T)
            # B += diag(sigma), avoid computing diag(sigma) which is dxd
            B.ravel().put(idx_diag_d, B.ravel().take(idx_diag_d)+sigma)
            # this quantity is used later for the log-likelihood
            # abs is there to avoid numerical errors when det < 0
            log_det_B = numx.log(abs(det(B)))
            # end the computation of B
            B = inv(B)

            ## other useful quantities
            trA_B = mult(A.T, B)
            trA_B_cov_mtx = mult(trA_B, cov_mtx)

            ##### E-step
            ## E_yyT = E(y_n y_n^T | x_n)
            E_yyT = - mult(trA_B, A) + mult(trA_B_cov_mtx, trA_B.T)
            # E_yyT += numx.eye(k)
            E_yyT.ravel().put(idx_diag_k, E_yyT.ravel().take(idx_diag_k)+1.)

            ##### M-step
            A = mult(trA_B_cov_mtx.T, inv(E_yyT))
            sigma = cov_diag - (mult(A, trA_B_cov_mtx)).diagonal()

            ##### log-likelihood
            trace_B_cov = (B*cov_mtx.T).sum()
            # this is actually likelihood/tlen.
            lhood = const - 0.5*log_det_B - 0.5*trace_B_cov
            if verbose:
                print('cycle', t, 'log-lhood:', lhood)

            ##### convergence criterion
            if base_lhood is None:
                base_lhood = lhood
            else:
                # convergence criterion
                if (lhood-base_lhood)<(1.+tol)*(old_lhood-base_lhood):
                    break
                if lhood < old_lhood:
                    # this should never happen
                    # it sometimes does, e.g. if the noise is extremely low,
                    # because of numerical rounding effects
                    warnings.warn(_LHOOD_WARNING, mdp.MDPWarning)
            old_lhood = lhood
            lhood_curve.append(lhood)

        self.tlen = tlen
        self.A = A
        self.mu = mu.reshape(1, d)
        self.sigma = sigma

        ## MAP matrix
        # compute B = (A A^T + Sigma)^-1
        B = mult(A, A.T).copy()
        B.ravel().put(idx_diag_d, B.ravel().take(idx_diag_d)+sigma)
        B = inv(B)
        self.E_y_mtx = mult(B.T, A)

        self.lhood = lhood_curve

    def _execute(self, x):
        return mult(x-self.mu, self.E_y_mtx)

    @staticmethod
    def is_invertible():
        return False

    def generate_input(self, len_or_y=1, noise=False):
        """Generate data from the prior distribution.

        If the training phase has not been completed yet, call stop_training.
        
        :param len_or_y: If integer, it specified the number of observation
            to generate. If array, it is used as a set of samples
            of the latent variables

        :param noise: If true, generation includes the estimated noise
        :type noise: bool
        
        :return: The generated data.
        :rtype: numpy.ndarray
        """

        self._if_training_stop_training()

        # set the output dimension if necessary
        if self.output_dim is None:
            # if the input_dim is not defined, raise an exception
            if self.input_dim is None:
                errstr = ("Number of input dimensions undefined. Inversion "
                          "not possible.")
                raise NodeException(errstr)
            self.output_dim = self.input_dim

        if isinstance(len_or_y, int):
            size = (len_or_y, self.output_dim)
            y = self._refcast(mdp.numx_rand.normal(size=size))
        else:
            y = self._refcast(len_or_y)
            self._check_output(y)

        res = mult(y, self.A.T)+self.mu
        if noise:
            ns = mdp.numx_rand.normal(size=(y.shape[0], self.input_dim))
            ns *= numx.sqrt(self.sigma)
            res += self._refcast(ns)
        return res
