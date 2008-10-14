from mdp import numx, numx_linalg, Cumulator, TrainingException
from mdp.utils import mult, symeig, nongeneral_svd

# some useful functions
sqrt = numx.sqrt

# search XXX for locations where future work is needed

#########################################################
#  Hessian LLE
#########################################################

class LLENode(Cumulator):
    """Perform a Locally Linear Embedding analysis on the data

    Internal Variables:
      self.data : the training data
      self.training_projection : the LLE projection of the training data
                                 (defined when training finishes)
      self.k : number of nearest neighbors to use
      self.desired_variance : variance limit used to compute
                             intrinsic dimensionality
                             
    Based on the algorithm outlined in 'An Introduction to Locally
    Linear Embedding' by L. Saul and S. Roewis, using improvements
    suggested in 'Locally Linear Embedding for Classification' by
    D. deRidder and R.P.W. Duin.

    Python implementation by:
      Jake Vanderplas, University of Washington
      vanderplas@astro.washington.edu
    """
    
    def __init__(self, k, r=None, svd=False, verbose=False,
                 input_dim=None, output_dim=None, dtype=None):
        """
        Keyword Arguments:

         k -- number of nearest neighbors to use
         r -- regularization constant; if None, r is automatically
              computed using the method presented in deRidder and Duin;
              this method involves solving one eigenvalue problem for
              every data point, and can slow down the algorithm
              If specified, it multiplies the trace of the local covariance
              matrix of the distances, as in Saul & Roweis (faster)
         svd -- if True, use SVD to compute the projection matrix;
                SVD is slower but more stable

         output_dim -- number of dimensions to output
                       or a float between 0.0 and 1.0
        """

        if output_dim <= 1 and isinstance(output_dim, float):
            self.desired_variance = output_dim
            output_dim = None
        else:
            self.desired_variance = None
         
        super(LLENode, self).__init__(input_dim, output_dim, dtype)

        self.k = k
        self.r = r
        self.svd = svd
        self.verbose = verbose
 
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _stop_training(self):
        super(LLENode, self)._stop_training()

        if self.verbose:
            msg = ('training LLE on %i points' 
                   ' in %i dimensions...' % (self.data.shape[0],
                                             self.data.shape[1]))
            print msg

        # some useful quantities
        M = self.data
        N = M.shape[0]
        k = self.k
        r = self.r

        # indices of diagonal elements
        W_diag_idx = numx.arange(N)
        Q_diag_idx = numx.arange(k)

        if k > N:
            err = ('LLENode: k=%i must be less than or ' 
                   'equal to number of training points N=%i' % (k,N))
            raise TrainingException(err)
 
        # determines number of output dimensions: if desired_variance
        # is specified, we need to learn it from the data. Otherwise,
        # it's easy
        learn_outdim = False
        if self.output_dim is None:
            if self.desired_variance is None:
                self.output_dim = self.input_dim
            else:
                learn_outdim = True

        # do we need to automatically determine the regularization term?
        auto_reg = r is None

        # determine number of output dims, precalculate useful stuff
        if learn_outdim:
            Qs, sig2s, nbrss = self._adjust_output_dim()

        # build the weight matrix
        #XXX future work:
        #XXX   for faster implementation, W should be a sparse matrix
        W = numx.zeros((N, N), dtype=self.dtype)

        if self.verbose:
            print ' - constructing [%i x %i] weight matrix...' % W.shape
    
        for row in range(N):
            if learn_outdim:
                Q = Qs[row, :, :]
                nbrs = nbrss[row, :]
            else:
                # -----------------------------------------------
                #  find k nearest neighbors
                # -----------------------------------------------
                M_Mi = M-M[row]
                nbrs = numx.argsort( (M_Mi**2).sum(1) )[1:k+1]
                M_Mi = M_Mi[nbrs]
                # compute covariance matrix of distances
                Q = mult(M_Mi, M_Mi.T)
                
            # -----------------------------------------------
            #  compute weight vector based on neighbors
            # -----------------------------------------------

            #Covariance matrix may be nearly singular:
            # add a diagonal correction to prevent numerical errors
            if auto_reg:
                # automatic mode: correction is equal to the sum of
                # the (d_in-d_out) unused variances (as in deRidder &
                # Duin)
                if learn_outdim:
                    sig2 = sig2s[row, :]
                else:
                    sig2 = (numx_linalg.svd(M_Mi, compute_uv=0))**2
                r = numx.sum(sig2[self.output_dim:])
                Q[Q_diag_idx, Q_diag_idx] += r
            else:
                # Roweis et al instead use "a correction that 
                #   is small compared to the trace" e.g.:
                # r = 0.001 * float(Q.trace())
                # this is equivalent to assuming 0.1% of the variance is unused
                Q[Q_diag_idx, Q_diag_idx] += r*Q.trace()
                
            #solve for weight
            # weight is w such that sum(Q_ij * w_j) = 1 for all i
            w = self._refcast(numx_linalg.solve(Q, numx.ones(k)))
            w /= w.sum()

            #update row of the weight matrix
            W[nbrs, row] = w

        if self.verbose:
            msg = (' - finding [%i x %i] null space of weight matrix\n'
                   '     (may take a while)...' % (self.output_dim, N))
            print msg

        self.W = W.copy()
        #to find the null space, we need the bottom d+1
        #  eigenvectors of (W-I).T*(W-I)
        #Compute this using the svd of (W-I):
        W[W_diag_idx, W_diag_idx] -= 1.

        #XXX future work:
        #XXX  use of upcoming ARPACK interface for bottom few eigenvectors
        #XXX   of a sparse matrix will significantly increase the speed
        #XXX   of the next step
        if self.svd:
            sig, U = nongeneral_svd(W.T, range=(2, self.output_dim+1))
        else:
            # the following code does the same computation, but uses
            # symeig, which computes only the required eigenvectors, and
            # is much faster. However, it could also be more unstable...
            WW = mult(W, W.T)
            # regularizes the eigenvalues, does not change the eigenvectors:
            WW[W_diag_idx, W_diag_idx] += 0.1
            sig, U = symeig(WW, range=(2, self.output_dim+1), overwrite=True)

        self.training_projection = U

    def _adjust_output_dim(self):
        # this function is called if we need to compute the number of
        # output dimensions automatically; some quantities that are
        # useful later are pre-calculated to spare precious time

        if self.verbose:
            print ' - adjusting output dim:'

        #otherwise, we need to compute output_dim
        #                  from desired_variance
        M = self.data
        k = self.k
        N, d_in = M.shape

        m_est_array = []
        Qs = numx.zeros((N, k, k))
        sig2s = numx.zeros((N, d_in))
        nbrss = numx.zeros((N, k), dtype='i')
        
        for row in range(N):
            #-----------------------------------------------
            #  find k nearest neighbors
            #-----------------------------------------------
            M_Mi = M-M[row]
            nbrs = numx.argsort( (M_Mi**2).sum(1) )[1:k+1]
            M_Mi = M_Mi[nbrs]
            # compute covariance matrix of distances
            Qs[row, :, :] = mult(M_Mi, M_Mi.T)
            nbrss[row, :] = nbrs

            #-----------------------------------------------
            # singular values of M_Mi give the variance:
            #   use this to compute intrinsic dimensionality
            #   at this point
            #-----------------------------------------------
            sig2 = (numx_linalg.svd(M_Mi, compute_uv=0))**2
            sig2s[row, :] = sig2

            #-----------------------------------------------
            # use sig2 to compute intrinsic dimensionality of the
            #   data at this neighborhood.  The dimensionality is the
            #   number of eigenvalues needed to sum to the total
            #   desired variance
            #-----------------------------------------------
            sig2 /= sig2.sum()
            S = sig2.cumsum()
            m_est = S.searchsorted(self.desired_variance)
            if m_est > 0:
                m_est += (self.desired_variance-S[m_est-1])/sig2[m_est]
            else:
                m_est = self.desired_variance/sig2[m_est]
            m_est_array.append(m_est)

        m_est_array = numx.asarray(m_est_array)
        self.output_dim = int( numx.ceil( numx.median(m_est_array) ) )
        if self.verbose:
            msg = ('      output_dim = %i'
                   ' for variance of %.2f' % (self.output_dim,
                                              self.desired_variance))
            print msg

        return Qs, sig2s, nbrss
    
    def _execute(self, x):
        #----------------------------------------------------
        # similar algorithm to that within self.stop_training()
        #  refer there for notes & comments on code
        #----------------------------------------------------
        N = self.data.shape[0]
        Nx = x.shape[0]
        W = numx.zeros((Nx, N), dtype=self.dtype)

        k, r = self.k, self.r
        d_out = self.output_dim
        Q_diag_idx = numx.arange(k)
        
        for row in range(Nx):
            #find nearest neighbors of x in M
            M_xi = self.data-x[row]
            nbrs = numx.argsort( (M_xi**2).sum(1) )[:k]
            M_xi = M_xi[nbrs]

            #find corrected covariance matrix Q
            Q = mult(M_xi, M_xi.T)
            if r is None and k > d_out:
                sig2 = (numx_linalg.svd(M_xi, compute_uv=0))**2
                r = numx.sum(sig2[d_out:])
                Q[Q_diag_idx, Q_diag_idx] += r
            if r is not None:
                Q[Q_diag_idx, Q_diag_idx] += r

            #solve for weights
            w = self._refcast(numx_linalg.solve(Q , numx.ones(k)))
            w /= w.sum()
            W[row, nbrs] = w

        #multiply weights by result of SVD from training
        return numx.dot(W, self.training_projection)
    
    def is_trainable(self):
        return True

    def is_invertible(self):
        return False


#########################################################
#  Hessian LLE
#########################################################


class HLLENode(LLENode):
    """Perform a Hessian Locally Linear Embedding analysis on the data
    
    Internal Variables:
      self.training_data : the training data

      self.training_projection : the HLLE projection of the training data
                                 (defined when training finishes)

      self.k : number of nearest neighbors to use

      self.desired_variance : variance limit used to compute
                             intrinsic dimensionality

      self.use_svd : use svd rather than eigenvalues of the covariance matrix
                            
    Implementation based on algorithm outlined in
     'Hessian Eigenmaps: new locally linear embedding techniques
      for high-dimensional data'
        by C. Grimes and D. Donoho, March 2003 

    Python implementation by:
      Jake Vanderplas, University of Washington
      vanderplas@astro.washington.edu
    """

    #----------------------------------------------------
    # Note that many methods ar inherited from LLENode,
    #  including _execute(), _adjust_output_dim(), etc.
    # The main advantage of the Hessian estimator is to
    #  limit distortions of the input manifold.  Once
    #  the model has been trained, it is sufficient (and
    #  much less computationally intensive) to determine
    #  projections for new points using the LLE framework.
    #----------------------------------------------------
    
    def __init__(self, k=None, input_dim=None, output_dim=None,
                 dtype=None, use_svd=True):
        self.use_svd = use_svd
        super(HLLENode, self).__init__(k, input_dim, output_dim, dtype)

    def _stop_training(self):
        M = self.training_data
        
        if self.verbose:
            print 'performing HLLE on %i points in %i dimensions...' % M.shape

        self._adjust_output_dim()
        k = self.k
        d_out = self.output_dim
        d_in = self.input_dim
        N = M.shape[0]
        
        if k > N:
            err = ('HLLENode: k=%i must be less than'
                   ' or equal to number of training points N=%i' % (k, N))
            raise TrainingException(err)

        #dp = d_out + (d_out-1) + (d_out-2) + ...
        dp = d_out*(d_out+1)/2

        #build the weight matrix
        #XXX future work:
        #XXX   for faster implementation, W should be a sparse matrix
        W = numx.asmatrix( numx.zeros((N, dp*N)) )

        if self.verbose:
            print ' - constructing [%i x %i] weight matrix...' % W.shape

        for row in range(N):
            #-----------------------------------------------
            #  find k nearest neighbors
            #-----------------------------------------------
            M_Mi = M-M[row]
            nbrs = numx.argsort( (M_Mi**2).sum(1) )[1:k+1]

            #-----------------------------------------------
            #  center the neighborhood using the mean
            #-----------------------------------------------
            nbrhd = M[nbrs]
            nbrhd -= nbrhd.mean(0)

            #-----------------------------------------------
            #  compute local coordinates
            #   using a singular value decomposition
            #-----------------------------------------------
            U, sig, VT = numx_linalg.svd(nbrhd, full_matrices=0)
            nbrhd = numx.asmatrix( U.T[:d_out] )
            del VT

            #-----------------------------------------------
            #  build Hessian estimator
            #-----------------------------------------------
            Yi = numx.asmatrix(numx.zeros([dp, k]))
            ct = 0
        
            for i in range(d_out):
                for j in range(i, d_out):
                    Yi[ct] = numx.multiply( nbrhd[i], nbrhd[j] )
                    ct += 1
            Yi = numx.concatenate( [numx.ones((1, k)), nbrhd, Yi], 0 )

            #-----------------------------------------------
            #  orthogonalize linear and quadratic forms
            #   with QR factorization
            #  and make the weights sum to 1
            #-----------------------------------------------
            Q, R = numx_linalg.qr(Yi.T)
            w = numx.asarray(Q[:, d_out+1:])
            S = w.sum(0) #sum along columns

            #if S[i] is too small, set it equal to 1.0
            # this prevents weights from blowing up
            S[numx.where(numx.absolute(S)<1E-4)] = 1.0
            W[ nbrs , row*dp:(row+1)*dp ] = w / S

        #-----------------------------------------------
        # To find the null space, we want the
        #  first d+1 eigenvectors of W.T*W
        # Compute this using an svd of W
        #-----------------------------------------------
        
        if self.verbose:
            msg = (' - finding [%i x %i] ' 
                   'null space of weight matrix...' % (d_out, N))
            print msg

        #XXX future work:
        #XXX  use of upcoming ARPACK interface for bottom few eigenvectors
        #XXX   of a sparse matrix will significantly increase the speed
        #XXX   of the next step
        
        #Fast, but memory intensive
        if self.use_svd:
            U, sig, VT = numx_linalg.svd(W, full_matrices=0)
            del VT
            del W
            indices = numx.argsort(sig)[1:d_out+1]
            Y = U[:, indices] * numx.sqrt(N)
            
        #Slower, but uses less memory
        else:
            C = W*W.T
            del W
            sig2, V = numx_linalg.eigh(C)
            del C
            indices = numx.argsort(sig)[1:d_out+1]
            Y = V[:, indices] * numx.sqrt(N)

        #-----------------------------------------------
        # Normalize Y
        #  we need R = (Y.T*Y)^(-1/2)
        #   do this with an SVD of Y
        #      Y = U*sig*V.T
        #      Y.T*Y = (V*sig.T*U.T) * (U*sig*V.T)
        #            = V * (sig*sig.T) * V.T
        #            = V * sig^2 V.T
        #   so
        #      R = V * sig^-1 * V.T
        #-----------------------------------------------
        if self.verbose:
            print ' - normalizing null space via SVD...'

        #Fast, but memory intensive
        if self.use_svd:
            U, sig, VT = numx_linalg.svd(Y, full_matrices=0)
            del VT
            S = numx.asmatrix(numx.diag(sig**-1))
            self.training_projection = numx.asarray(U * S * U.T * Y)

        #Slower, but uses less memory
        else:
            C = Y.T*Y
            sig2, U = numx_linalg.eigh(C)
            U = U[:, ::-1] #eigenvectors should be in descending order
            sig2 = sig2[::-1]
            S = numx.asmatrix(numx.diag( (1.0*sig2)**-1.5))
            self.training_projection = numx.asarray(Y * U * S * U.T * C)
    
