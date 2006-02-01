import mdp
from mdp import numx, numx_linalg, utils
from mdp.utils import mult, normal
from lcov import CovarianceMatrix, DelayCovarianceMatrix
import warnings

take, put, diag, ravel = numx.take, numx.put, utils.diag, numx.ravel
sqrt, tr, inv, det = numx.sqrt, numx.transpose, utils.inv, utils.det

# decreasing likelihood message
_LHOOD_WARNING = 'Likelihood decreased in FANode. This is probably due '+\
                 'to some numerical errors.'
warnings.filterwarnings('always', _LHOOD_WARNING, mdp.MDPWarning)


class FANode(mdp.Node):

    def __init__(self, tol=1e-4, max_cycles=100, verbose=False,
                 input_dim=None, output_dim=None, typecode=None):
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
        typ = self.typecode
        one = self._scast(1.)
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
        const = self._scast(-d/2. * numx.log(2.*numx.pi))

        ##### request the covariance matrix and clean up
        cov_mtx, mu, tlen = self._cov_mtx.fix()
        del self._cov_mtx
        # 'bias' the covariance matrix
        # (i.e., cov_mtx = 1/tlen sum(x_t x_t^T) instead of 1/(tlen-1))
        cov_mtx *= (tlen-self._scast(1.))/tlen
        cov_diag = diag(cov_mtx).astype(typ)

        ##### initialize the parameters
        # noise variances
        sigma = cov_diag
        # loading factors
        # Zoubin uses the determinant of cov_mtx^1/d as scale but it's
        # too slow for large matrices. Ss the product of the diagonal a good
        # approximation?
        #scale = det(cov_mtx)**(1./d)
        scale = numx.product(sigma)**(1./d)
        A = normal(0., sqrt(scale/k), shape=(d,k)).astype(typ)

        ##### EM-cycle
        lhood_curve = []
        base_lhood = None
        old_lhood = -utils.inf
        for t in xrange(self.max_cycles):
            ## compute B = (A A^T + Sigma)^-1
            # copy to make it contiguous, otherwise .flat does not work
            B = mult(A, tr(A)).copy()
            # B += diag(sigma), avoid computing diag(sigma) which is dxd
            put(B.flat, idx_diag_d, take(B.flat, idx_diag_d)+sigma)
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
            put(E_yyT.flat, idx_diag_k, take(E_yyT.flat, idx_diag_k)+one)
            
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
        put(B.flat, idx_diag_d, take(B.flat, idx_diag_d)+sigma)
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
            ns = utils.normal(0., self.sigma,
                              shape=(y.shape[0], self.input_dim))
            res += self.refcast(ns)
        return res


class KalmanNode(mdp.Node):
    
    def _get_train_seq(self):
        return [(self._train_init1, self._stop_init1),
                (self._train_init2, self._stop_init2),
                (self._train_em, self._stop_em)]

    def __init__(self, output_dim, input_dim=None, typecode=None):
        """State Space Model / Linear Dynamical System / Kalman Filter

        This node is a basic implementation of the EM algorithm for
        Linear Dynamical Systems. It could probably be refined to save
        some memory and time.

        output_dim -- number of latent variables
        """
        # Notation as in Gaharamani and Hinton, 1996
        super(KalmanNode, self).__init__(input_dim, output_dim, typecode)
        
        # x(t+1) = Ax(t) + w(t)
        self.A = None
        # y(t) = Cx(t) + v(t)
        self.C = None
        
        # ?? zoubin chooses R and Q to be diagonal, which is restricitve
        # but is more efficient. It certainly saves a lot of memory.
        # covariance of w(t)
        self.Q = None
        # covariance of v(t)
        self.R = None
        # mean of v(1)
        self.pi1 = None
        # covariance of v(1)
        self.V1 = None
        # mean of the observations
        self.y_mean = None

    ### Kalman filter
        
    def filter(self, y, _internals=False):
        # some definitions
        A = self.A
        tr_A = tr(A) # reference
        C = self.C
        tr_C = tr(C) # reference
        R = self.R
        Q = self.Q
        k = self.output_dim
        tlen = y.shape[0]

        # ?? save _t1 variables only when internals is set
        # E(xt | y1 ... yt-1)
        xt_t1 = []
        # E(xt | y1 ... yt)
        xt_t = []
        # Var(xt | y1 ... yt-1)
        Vt_t1 = []
        # Var(xt | y1 ... yt)
        Vt_t = []
        # Kalman gain
        if _internals:
            Kt = []
            # constant term in front of the log-likelihood
            const = self._scast(-self.input_dim/2. * numx.log(2.*numx.pi))
            # log-likelihood
            lhood = 0.

        # forward recursion
        xpre = self.pi1
        Vpre = self.V1
        for i in xrange(tlen):
            xt_t1.append(xpre)
            Vt_t1.append(Vpre)

            # ?? zoubin computes Kt in two different ways depending on
            # the dimensionalities
            
            # useful quantity
            VC = mult(Vpre, tr_C)
            ## compute the Kalman gain K
            Sigma = mult(C, VC) + R
            # this quantity is used later for the log-likelihood
            # abs is there to avoid numerical errors when det < 0 
            log_det_Sigma = numx.log(abs(det(Sigma)))
            inv_Sigma = inv(Sigma)
            K = mult(VC, inv_Sigma)

            # ?? allow for missing values by putting xpost=xpre, Vpost=Vpre
            # new estimation
            # xpost := E(xt+1 | y1 ... yt+1)
            y_diff = y[i:i+1,:] - mult(xpre, tr_C)
            xpost = xpre + mult(y_diff, tr(K))
            # Vpost := Var(xt+1 | y1 ... yt+1)
            # this is the equation in Gaharamani and Hinton:
            # Vpost = Vpre - mult(K, tr(VC))
            # the following way is much more stable (see M.Welling eq. 43)
            # ?? should I get rid of eye?
            I_KC = utils.eye(k) - mult(K, C)
            Vpost = mult(I_KC, mult(Vpre, tr(I_KC))) + mult(K, mult(R, tr(K)))

            # ?? I think I'm losing the last point
            if _internals:
                lhood += float(const - 0.5*log_det_Sigma \
                               - 0.5*mult(y_diff, mult(inv_Sigma, tr(y_diff))))

            # ?? do not compute this on the last iteration
            # xpre := E(xt+1 | y1 ... yt)
            xpre = mult(xpost, tr_A)
            # Vpre := Var(xt+1 | y1 ... yt)
            Vpre = mult(A, mult(Vpost, tr_A)) + Q

            xt_t.append(xpost)
            Vt_t.append(Vpost)
            if _internals: Kt.append(K)
                
        if _internals:
            return xt_t, Vt_t, xt_t1, Vt_t1, Kt, lhood
        else:
            xt_t = numx.reshape(numx.asarray(xt_t, typecode=self.typecode),
                                (tlen, k))
            Vt_t = numx.reshape(numx.asarray(Vt_t, typecode=self.typecode),
                                (tlen, k, k))
            return xt_t, Vt_t

    ### Kalman smoother

    def smooth(self, y, _internals=False):
        xt_t, Vt_t, xt_t1, Vt_t1, Kt, lhood = self.filter(y, _internals=True)
        if not _internals: del Kt, lhood

        # ?? to save memory one could overwrite xt_t and delete xt_t1 and Kt
        # during the recursion
        
        # some definitions
        A = self.A
        tr_A = tr(A) # reference
        C = self.C
        R = self.R
        Q = self.Q
        k = self.output_dim
        tlen = y.shape[0]

        J = [None]*(tlen-1)
        # E(xt | y1 ... yT)
        xt_T = [None]*tlen
        # Var(xt | y1 ... yT)
        Vt_T = [None]*tlen

        # backward recursion
        xt_T[-1] = xt_t[-1]
        Vt_T[-1] = Vt_t[-1]
        for i in xrange(tlen-1, 0, -1):
            J[i-1] = mult(Vt_t[i-1], mult(tr(A), inv(Vt_t1[i])))
            # ?? ... + mult(ct_T[i]-xt_t1[i], tr(J[i-1]))
            xt_T[i-1] = xt_t[i-1] + \
                        mult(xt_T[i]-mult(xt_t[i-1], tr_A), tr(J[i-1]))
            Vt_T[i-1] = Vt_t[i-1] + \
                        mult(J[i-1], mult(Vt_T[i]-Vt_t1[i], tr(J[i-1])))

        # additional quantities of interest for EM, computed only if requested
        if _internals:
            Vt_t1_T = [None]*tlen
            k = self.output_dim
            # ?? is it interesting to get rid of eye?
            Vt_t1_T[-1] = mult(mult(utils.eye(k)-mult(Kt[-1], C), A), Vt_t[-2])
            
            for i in xrange(tlen-1, 1, -1):
                tmp = Vt_t1_T[i] - mult(A, Vt_t[i-1])
                Vt_t1_T[i-1] = mult(Vt_t[i-1], tr(J[i-2])) + \
                               mult(J[i-1], mult(tmp, tr(J[i-2])))
            
        if _internals:
            xt_T = numx.reshape(numx.asarray(xt_T, typecode=self.typecode),
                                (tlen, k))
            # ?? Vt_T =numx.squeeze(numx.asarray(Vt_T, typecode=self.typecode))
            return xt_T, Vt_T, Vt_t1_T, lhood
        else:
            xt_T = numx.reshape(numx.asarray(xt_T, typecode=self.typecode),
                                (tlen, k))
            Vt_T = numx.reshape(numx.asarray(Vt_T, typecode=self.typecode),
                                (tlen, k, k))
            return xt_T, Vt_T

    ######## EM training

    ### training phase 1: init Factor Analysis or AR1

    def _train_init1(self, y):
        return
        k = self._output_dim
        d = self._input_dim
        
        if not hasattr(self, 'init_node'):
            if k<=d:
                # initialize using Factor Analysis
                self.init_node = FANode(output_dim = k)
            else:
                raise NotImplementedException, 'k>d'

        self.init_node.train(y)

    def _stop_init1(self):
        return
        self.init_node.stop_training()

    ### training phase 2:
    ###    init internals using the Factor Analysis or AR1 estimate
    
    def _train_init2(self, y):
        return
        if not hasattr(self, 'x_cov'):
            self.x_cov = CovarianceMatrix(self.typecode)
            self.x_dcov = DelayCovarianceMatrix(1, self.typecode)
        x = self.init_node.execute(y)
        self.x_cov.update(x)
        self.x_dcov.update(x)
        
    def _stop_init2(self):
        return
        init_node = self.init_node
        k = self.output_dim
        d = self.input_dim
        type = self.typecode
        
        # request the covariance matrix and clean up
        x_cov, x_mean, tlen = self.x_cov.fix()
        del self.x_cov
        x_dcov, tmp, tmp, tlen = self.x_dcov.fix()
        del self.x_dcov, tmp

        # FA case, init internals
        self.y_mean = init_node.mu
        self.C = init_node.A
        self.R = diag(init_node.sigma)
        self.pi1 = x_mean
        self.V1 = x_cov
        self.Q = x_cov
        # ?? zoubin uses A=inv(x_cov+Q)*x_dcov
        self.A = mult(inv(x_cov), x_dcov)
        
        del self.init_node

    ### training phase 3: EM estimation
    def _train_em(self, y):
        #?? - y_mean
        y -= self.y_mean
        
        # ?? cycle until convergence
        for j in range(3):
            print
            print self.A
            print self.C
        
            # do one EM cycle using only the current data
            
            # E-step:estimate the latent variables using the current parameters
            x, Pt, Ptt1, lhood = self.smooth(y, _internals=True)
            #print x[:,0]
            print j, lhood

            # useful quantities
            tlen = y.shape[0]
            i=2
            # ?? better in smoother? useful at all?
            Pt = [Pt[i]+mult(tr(x[i:i+1,:]),x[i:i+1,:])
                  for i in range(1, tlen)]
            sum_1_T_P = sum(Pt)
            sum_2_T_P = sum_1_T_P - Pt[0]
            sum_1_T1_P = sum_1_T_P - Pt[-1]
            # ?? del Pt
            # ?? better in smoother? useful at all?
            Ptt1 = [Ptt1[i]+mult(tr(x[i:i+1,:]),x[i-1:i,:])
                    for i in range(1, tlen)]
            sum_2_T_Ptt1 = sum(Ptt1[1:])
            del Ptt1
            yx = mult(tr(y), x)
            print '\n###########\n', yx, '\n###########\n'

            ## M-step: update the parameters
            C = mult(yx, inv(sum_1_T_P))
            R = 1./tlen * (mult(tr(y), y) - mult(C, tr(yx)))
            A = mult(sum_2_T_Ptt1, inv(sum_1_T1_P))
            Q = 1./(tlen-1) * (sum_2_T_P - mult(A, sum_2_T_Ptt1))
            # ?? replace with the mean over all sequences
            pi1 = x[0:1,:]
            # ?? replace with the mean over all sequences
            V1 = Pt[0] - mult(tr(pi1), pi1)

            self.C, self.R, self.A, self.Q, self.pi1, self.V1 = \
                    C, R, A, Q, pi1, V1

    def _stop_em(self):
        pass

    def _execute_filter(self, y):
        return self.filter(y-self.y_mean)[0]
    
    def _execute_smooth(self, y):
        return self.smooth(y-self.y_mean)[0]

    def _execute(self, y):
        return self._execute_smooth(y)

    def _inverse(self, x):
        # ?? noise
        return mult(x, tr(self.C))+self.y_mean
