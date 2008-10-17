from mdp import numx, numx_linalg, numx_rand, utils, \
    Node, Cumulator, TrainingException, MDPWarning
from mdp.utils import mult, symeig, nongeneral_svd
import warnings as _warnings

# some useful functions
sqrt = numx.sqrt

# search XXX for locations where future work is needed

#########################################################
#  Locally Linear Embedding
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
    
    def __init__(self, k, r=0.001, svd=False, verbose=False,
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

        if isinstance(output_dim, float) and output_dim <= 1:
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
        Cumulator._stop_training(self)

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
            err = ('k=%i must be less than or ' 
                   'equal to number of training points N=%i' % (k, N))
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
                nbrs = numx.argsort((M_Mi**2).sum(1))[1:k+1]
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
                    sig2 = utils.svd(M_Mi, compute_uv=0)**2
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
            # XXX refcast is due to numpy bug: floats become double
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
            nbrs = numx.argsort((M_Mi**2).sum(1))[1:k+1]
            M_Mi = M_Mi[nbrs]
            # compute covariance matrix of distances
            Qs[row, :, :] = mult(M_Mi, M_Mi.T)
            nbrss[row, :] = nbrs

            #-----------------------------------------------
            # singular values of M_Mi give the variance:
            #   use this to compute intrinsic dimensionality
            #   at this point
            #-----------------------------------------------
            sig2 = (utils.svd(M_Mi, compute_uv=0))**2
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
                sig2 = (utils.svd(M_xi, compute_uv=0))**2
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


# Modified Gram-Schmidt
def _mgs(a):
    m, n = a.shape
    v = a.copy()
    r = numx.zeros((n, n))
    for i in range(n):
        r[i,i] = numx_linalg.norm(v[:,i])
        v[:,i] = v[:,i]/r[i,i]
        for j in range(i+1, n):
            r[i,j] = mult(v[:,i], v[:,j])
            v[:,j] = v[:,j] - r[i,j]*v[:,i]
    # q is v
    return v, r

class HLLENode(LLENode):
    """Perform a Hessian Locally Linear Embedding analysis on the data
    
    Internal Variables:
      self.training_data : the training data

      self.training_projection : the HLLE projection of the training data
                                 (defined when training finishes)

      self.k : number of nearest neighbors to use; we recommend to
               choose k>=1 + output_dim + output_dim*(output_dim+1)/2,
               so that one can use a more efficient computing method

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
    
    def __init__(self, k, r=0.001, svd=False, verbose=False,
                 input_dim=None, output_dim=None, dtype=None):
        LLENode.__init__(self, k, r, svd, verbose,
                         input_dim, output_dim, dtype)

    def _stop_training(self):
        Cumulator._stop_training(self)
        
        k = self.k
        M = self.data
        N = M.shape[0]
        
        if k > N:
            err = ('k=%i must be less than'
                   ' or equal to number of training points N=%i' % (k, N))
            raise TrainingException(err)
        
        if self.verbose:
            print 'performing HLLE on %i points in %i dimensions...' % M.shape

        # determines number of output dimensions: if desired_variance
        # is specified, we need to learn it from the data. Otherwise,
        # it's easy
        learn_outdim = False
        if self.output_dim is None:
            if self.desired_variance is None:
                self.output_dim = self.input_dim
            else:
                learn_outdim = True

        # determine number of output dims, precalculate useful stuff
        if learn_outdim:
            Qs, sig2s, nbrss = self._adjust_output_dim()

        d_out = self.output_dim
        d_in = self.input_dim

        #dp = d_out + (d_out-1) + (d_out-2) + ...
        dp = d_out*(d_out+1)/2

        if min(k,N) <= d_out:
            err = ('k=%i and n=%i (number of input data points) must be'
                   ' larger than output_dim=%i' % (k, N, d_out))
            raise TrainingException(err)
        
        if k < 1+d_out+dp:
            wrn = ('The number of neighbours, k=%i, is smaller than'
                   ' 1 + output_dim + output_dim*(output_dim+1)/2 = %i,'
                   ' which might result in unstable results.'
                   % (k, 1+d_out+dp))
            _warnings.warn(wrn, MDPWarning)

        #build the weight matrix
        #XXX   for faster implementation, W should be a sparse matrix
        W = numx.zeros((N, dp*N), dtype=self.dtype)

        if self.verbose:
            print ' - constructing [%i x %i] weight matrix...' % W.shape

        for row in range(N):
            if learn_outdim:
                nbrs = nbrss[row,:]
            else:
                # -----------------------------------------------
                #  find k nearest neighbors
                # -----------------------------------------------
                M_Mi = M-M[row]
                nbrs = numx.argsort((M_Mi**2).sum(1))[1:k+1]

            #-----------------------------------------------
            #  center the neighborhood using the mean
            #-----------------------------------------------
            nbrhd = M[nbrs] # this makes a copy
            nbrhd -= nbrhd.mean(0)

            #-----------------------------------------------
            #  compute local coordinates
            #   using a singular value decomposition
            #-----------------------------------------------
            U, sig, VT = utils.svd(nbrhd)
            nbrhd = U.T[:d_out]
            del VT
            
            #-----------------------------------------------
            #  build Hessian estimator
            #-----------------------------------------------
            Yi = numx.zeros((dp, k), dtype=self.dtype)
            ct = 0
            for i in range(d_out):
                Yi[ct:ct+d_out-i,:] = nbrhd[i] * nbrhd[i:,:]
                ct += d_out-i
            Yi = numx.concatenate([numx.ones((1, k), dtype=self.dtype),
                                   nbrhd, Yi], 0)
            
            #-----------------------------------------------
            #  orthogonalize linear and quadratic forms
            #   with QR factorization
            #  and make the weights sum to 1
            #-----------------------------------------------
            if k >= 1+d_out+dp:
                Q, R = numx_linalg.qr(Yi.T)
                w = Q[:,d_out+1:d_out+1+dp]
            else:
                q, r = _mgs(Yi.T)
                w = q[:,-dp:]
            
            S = w.sum(0) #sum along columns
            #if S[i] is too small, set it equal to 1.0
            # this prevents weights from blowing up
            S[numx.where(numx.absolute(S)<1E-4)] = 1.0
            #print w.shape, S.shape, (w/S).shape
            #print W[nbrs, row*dp:(row+1)*dp].shape
            W[nbrs, row*dp:(row+1)*dp] = w / S

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
        
        if self.svd:
            sig, U = nongeneral_svd(W.T, range=(2, d_out+1))
            Y = U*numx.sqrt(N)         
        else:
            WW = mult(W, W.T)
            # regularizes the eigenvalues, does not change the eigenvectors:
            W_diag_idx = numx.arange(N)
            WW[W_diag_idx, W_diag_idx] += 0.01
            sig, U = symeig(WW, range=(2, self.output_dim+1), overwrite=True)            
            Y = U*numx.sqrt(N)
            del WW
        del W

        #-----------------------------------------------
        # Normalize Y
        #
        # Alternative way to do it:
        #  we need R = (Y.T*Y)^(-1/2)
        #   do this with an SVD of Y            del VT

        #      Y = U*sig*V.T
        #      Y.T*Y = (V*sig.T*U.T) * (U*sig*V.T)
        #            = V * (sig*sig.T) * V.T
        #            = V * sig^2 V.T
        #   so
        #      R = V * sig^-1 * V.T
        # The code is:
        #    U, sig, VT = utils.svd(Y)
        #    del U
        #    S = numx.diag(sig**-1)
        #    self.training_projection = mult(Y, mult(VT.T, mult(S, VT)))
        #-----------------------------------------------
        if self.verbose:
            print ' - normalizing null space...'

        C = utils.sqrtm(mult(Y.T, Y))
        self.training_projection = mult(Y, C)

# from mdp import numx, numx_linalg, numx_rand
# from mdp.utils import mult, symeig, nongeneral_svd
# import pylab
# from matplotlib import ticker, axes3d

# #################################################
# # Testing Functions
# #################################################

# def S(theta):
#     """
#     returns x,y
#       a 2-dimensional S-shaped function
#       for theta ranging from 0 to 1
#     """
#     t = 3*numx.pi * (theta-0.5)
#     x = numx.sin(t)
#     y = numx.sign(t)*(numx.cos(t)-1)
#     return x,y

# def rand_on_S(N,sig=0,hole=False):
#     t = numx_rand.random(N)
#     x,z = S(t)
#     y = numx_rand.random(N)*5.0
#     if sig:
#         x += numx_rand.normal(scale=sig,size=N)
#         y += numx_rand.normal(scale=sig,size=N)
#         z += numx_rand.normal(scale=sig,size=N)
#     if hole:
#         indices = numx.where( ((0.3>t) | (0.7<t)) | ((1.0>y) | (4.0<y)) )
#         #indices = numx.where( (0.3>t) | ((1.0>y) | (4.0<y)) )
#         return x[indices],y[indices],z[indices],t[indices]
#     else:
#         return x,y,z,t

# def scatter_2D(x,y,t=None,cmap=pylab.cm.jet):
#     #fig = pylab.figure()
#     pylab.subplot(212)
#     if t==None:
#         pylab.scatter(x,y)
#     else:
#         pylab.scatter(x,y,c=t,cmap=cmap)

#     pylab.xlabel('x')
#     pylab.ylabel('y')


# def scatter_3D(x,y,z,t=None,cmap=pylab.cm.jet):
#     fig = pylab.figure

#     if t==None:
#         ax.scatter3D(x,y,z)
#     else:
#         ax.scatter3D(x,y,z,c=t,cmap=cmap)

#     if x.min()>-2 and x.max()<2:
#         ax.set_xlim(-2,2)
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
    
#     # elev, az
#     ax.view_init(10, -80)


# def runtest1(N=1000,k=15,r=None,sig=0,output_dim=0.9,hole=False,type='LLE',svd=False):
#     #generate data
#     x,y,z,t = rand_on_S(N,sig,hole=hole)
#     data = numx.asarray([x,y,z]).T

#     #train LLE and find projection
#     if type=='HLLE':
#         LN = HLLENode(k=k, output_dim=output_dim, verbose=True, svd=False)
#     else:
#         LN = LLENode(k=k, r=r, output_dim=output_dim, verbose=True, svd=True)
#     LN.train(data)
#     LN.stop_training()
#     projection = LN.training_projection
#     #projection = LN.execute(data)

#     #plot input in 3D
#     fig = pylab.figure(1, figsize=(6,8))
#     pylab.clf()
#     ax = axes3d.Axes3D(fig,rect=[0,0.5,1,0.5])
#     ax.scatter3D(x,y,z,c=t,cmap=pylab.cm.jet)
#     ax.set_xlim(-2,2)
#     ax.view_init(10, -80)

#     #plot projection in 2D
#     pylab.subplot(212)
#     pylab.scatter(projection[:,0],\
#                   projection[:,1],\
#                   c=t,cmap=pylab.cm.jet)

# #######################################
# #  Run Tests
# #######################################
# if __name__ == '__main__':
#     #runtest1(N=5000,k=12,sig=0,output_dim=2,hole=False,type='HLLE')
#     runtest1(N=500,k=7,sig=0,output_dim=2,hole=True,type='HLLE')
#     pylab.show()
