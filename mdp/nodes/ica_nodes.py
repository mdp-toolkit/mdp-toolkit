import math
import mdp

# import numeric module (scipy, Numeric or numarray)
numx, numx_rand = mdp.numx, mdp.numx_rand

utils = mdp.utils
mult = utils.mult
t = numx.transpose

class ICANode(mdp.Cumulator, mdp.Node):
    """
    ICANode is a general class to handle different batch-mode algorithm for
    Independent Component Analysis. More information about ICA can be found
    among others in
    Hyvarinen A., Karhunen J., Oja E., Independent Component Analysis,
    Wiley (2001)."""
    
    def __init__(self, limit = 0.001, telescope = 0, verbose = 0, \
                 whitened = 0, white_comp = None, input_dim = None, \
                 typecode= None):
        """
        - Set whitened == 1 if input data are already whitened.
          Otherwise the node will whiten the data itself.
          When whitened == 0, you can set 'white_comp' to the number
          of whitened components to keep during the calculation.
        - limit is the convergence threshold.
        - if telescope == 1, use the Telescope mode. Instead of using all
          inputs try larger and larger chunks of the input data until
          convergence is achieved. This should lead to significantly faster
          convergence for stationary statistics.
          This mode has not been thoroughly tested and must be considered beta.
        """
        super(ICANode, self).__init__(input_dim, None, typecode)
        self.telescope = telescope
        self.verbose = verbose
        self.limit = limit
        self.whitened = whitened
        self.white_comp = white_comp

    def get_supported_typecodes(self):
        return ['f','d']

    def _stop_training(self):
        """Whiten data if needed and call the 'core' routine to perform ICA.
           Take care of telescope-mode if needed.
        """
        super(ICANode, self)._stop_training()
        
        verbose = self.verbose
        core = self.core
        limit = self.limit

        # ?? rewrite as a 2-phases node
        # whiten if needed
        if not self.whitened:
            self.output_dim = self.white_comp
            white = mdp.nodes.WhiteningNode(output_dim = self.white_comp)
            white.train(self.data)
            self.data = white.execute(self.data)
            self.white = white

        data = self.data
        
        # call 'core' in telescope mode if needed
        if self.telescope:
            minpow = math.frexp(self.input_dim*10)[1]
            maxpow = int(numx.log(data.shape[0])/numx.log(2))
            for tel in range(minpow,maxpow+1):
                index = 2**tel
                if verbose: print "--\nUsing %d inputs" %index
                convergence = core(data[:index,:])
                if convergence <= limit:
                    break
        else:
            convergence = core(data)
        if verbose: print "Convergence criterium: ", convergence
        self.convergence = convergence

    def core(self,data):
        """This is the core routine of the ICANode. Each subclass must
        define this function to return the achieved convergence value.
        This function is also responsible for setting the ICA filters
        matrix self.filters.
        Note that the matrix self.filters is applied to the right of the
        matrix containing input data. This is the transposed of the matrix
        defining the linear transformation."""
        pass

    def _execute(self, x):
        if not self.whitened:
            x = self.white.execute(x)
        return mult(x,self.filters)

    def _inverse(self, y):
        y = mult(y,t(self.filters))
        if not self.whitened:
            y = self.white.inverse(y)
        return y

class CuBICANode(ICANode):
    """
    CuBICANode receives an input signal and performs Independent Component
    Analysis using the CuBICA algorithm by Tobias Blaschke (see reference
    below). Note that CuBICA is a batch-algorithm. This means that it needs
    all input data before it can start and compute the ICs.
    The algorithm is here given as a Node for convenience, but it
    actually accumulates all inputs it receives. Remember that to avoid
    running out of memory when you have many components and many time samples.
    
    Reference:
    Blaschke, T. and Wiskott, L. (2003).
    CuBICA: Independent Component Analysis by Simultaneous Third- and
    Fourth-Order Cumulant Diagonalization.
    IEEE Transactions on Signal Processing, 52(5), pp. 1250--1256."""

    def core(self,data):
        # keep track of maximum angle of rotation
        # angles vary in the range [-pi, +pi]
        # put here -2pi < -pi < +pi
        self.maxangle = [-2*numx.pi]
        verbose = self.verbose
        telescope = self.telescope

        # we need to copy to avoid overwriting during rotation.
        x = data.copy()

        # convergence criterium == maxangle
        limit = self.limit
        comp = x.shape[1]
        tlen = self._scast(x.shape[0])
        
        # some casted constants (this is due to casting issues in the
        # numeric libraries. they are going to disappear as soon as
        # new casting conventions are established -> Numeric 3,
        # scipy 3.3, ...)
        scalars = numx.arange(0, 37, typecode=self.typecode)
        ct_c34 = self._scast(0.0625)
        ct_s34 = self._scast(0.25)
        ct_c44 = self._scast(1./384)
        ct_s44 = self._scast(1./96)

        # initial transposed rotation matrix == identity matrix
        Qt = numx.identity(comp, typecode=self.typecode)

        # maximum number of sweeps through all possible pairs of signals
        num = int(1+round(numx.sqrt(comp)))
        if verbose and not telescope: prog = utils.ProgressBar(0, num*(comp-1))
        count = 0

        # start sweeping 
        for k in range(num):
            maxangle = 0
            for i in range(comp - 1):
                for j in range(i+1,comp):
                    u1 = x[:,i]
                    u2 = x[:,j]
                    sq1 = x[:,i]*x[:,i]
                    sq2 = x[:,j]*x[:,j]

                    # calculate the cumulants of 3rd and 4th order.
                    C111  = mult(sq1,u1)/tlen
                    C112  = mult(sq1,u2)/tlen
                    C122  = mult(sq2,u1)/tlen
                    C222  = mult(sq2,u2)/tlen
                    C1111 = mult(sq1,sq1)/tlen - scalars[3]
                    C1112 = mult(sq1*u1,u2)/tlen
                    C1122 = mult(sq1,sq2)/tlen - scalars[1]
                    C1222 = mult(sq2*u2,u1)/tlen
                    C2222 = mult(sq2,sq2)/tlen - scalars[3]
                                        
                    c_34 = ct_c34 * (            (C111*C111+C222*C222)-
                                      scalars[3]*(C112*C112+C122*C122)-
                                      scalars[2]*(C111*C122+C112*C222)  )
                    s_34 = ct_s34 * (             C111*C112-C122*C222   )
                    c_44 = ct_c44 *( scalars[7]*(C1111*C1111+C2222*C2222)-
                                     scalars[16]*(C1112*C1112+C1222*C1222)-
                                     scalars[12]*(C1111*C1122+C1122*C2222)-
                                     scalars[36]*(C1122*C1122)-
                                     scalars[32]*(C1112*C1222)-
                                      scalars[2]*(C1111*C2222)              )
                    s_44 = ct_s44 *( scalars[7]*(C1111*C1112-C1222*C2222)+
                                     scalars[6]*(C1112*C1122-C1122*C1222)+
                                                (C1111*C1222-C1112*C2222)  )

                    # rotation angle that maximize the contrast function
                    phi_max = -0.25 * numx.arctan2(s_34+s_44,c_34+c_44)

                    # get the new rotation matrix.
                    # Using the function rotate with angle 'phi' on
                    # a transformation matrix corresponds to the
                    # right-multiplication by a rotation matrix
                    # with angle '-phi'.
                    utils.rotate(Qt, phi_max, [i,j])

                    # rotate input data
                    utils.rotate(x, phi_max, [i,j])

                    # keep track of maximum angle of rotation
                    maxangle = max(maxangle, abs(float(phi_max)))
                    
                count += 1
                if verbose and not telescope: prog.update(count)
            self.maxangle.append(maxangle)
            if maxangle <= limit:
                break
        
        self.iter = k           
        if verbose:
            print "\nSweeps: ",k
        self.filters = Qt

        # return the convergence criterium
        return maxangle 

class FastICANode(ICANode):
    """
    FastICANode receives an input signal and performs Independent Component
    Analysis using the FastICA algorithm by Aapo Hyvarinen (see reference
    below). Note that FastICA is a batch-algorithm. This means that it needs
    all input data before it can start and compute the ICs.
    The algorithm is here given as a Node for convenience, but it
    actually accumulates all inputs it receives. Remember that to avoid
    running out of memory when you have many components and many time samples.
    
    Reference:
    Aapo Hyvarinen
    Fast and Robust Fixed-Point Algorithms for Independent Component Analysis
    IEEE Transactions on Neural Networks, 10(3):626--634, 1999.

    Note:
    - 1.4.1998  created for Matlab by Jarmo Hurri, Hugo Gavert,
                Jaakko Sarela, and Aapo Hyvarinen
    - 7.3.2003  modified for Python by Thomas Wendler
    - 3.6.2004  rewritten and adapted for scipy and MDP by MDP's authors
    - 25.5.2005 now independent from scipy. Requires Numeric or numarray"""
    
    def __init__(self, approach = 'defl', g = 'pow3', \
                 fine_tanh = 10, fine_gaus = 1, max_it = 1000, failures = 5,\
                 limit = 0.001, verbose = 0, whitened = 0, white_comp = None,\
                 input_dim = None, typecode=None):
        """
        - approach:          the approach to use. Possible values are:
                               'defl' --> deflation
                               'symm' --> symmetric
        - g:                 the nonlinearity to use. Possible values are
                             'pow3','tanh' or 'gaus'
        - fine_tanh:         parameter for fine-tuning of 'tanh'
        - fine_gaus:         parameter for fine-tuning of 'gaus'
        - max_it:            maximum number of iterations
        - failures           maximum number of failures to allow in
                             deflation mode

        Note: FastICA does not support the telescope mode (the convergence
              criterium is not robust in telescope mode)."""
        ICANode.__init__(self, limit, 0, verbose, whitened,\
                         white_comp, input_dim, typecode)
        if approach in ['defl','symm']:
            self.approach = approach
        else:
            raise mdp.NodeException, \
                  '%s approach method not known' % approach
        self.g = g
        self.fine_tanh = fine_tanh
        self.fine_gaus = fine_gaus
        self.max_it = max_it
        self.failures = failures

    def core(self,data):
        # Default values and initial definitions
        fine_tanh = self.fine_tanh
        fine_gaus = self.fine_gaus
        approach = self.approach
        g = self.g
        limit = self.limit
        max_it = self.max_it
        failures = self.failures
        typecode = self.typecode
        verbose = self.verbose
        X = t(data)
        
        # casted constants
        comp = X.shape[0]
        tlen = self._scast(X.shape[1])
        three = self._scast(3)
        one = self._scast(1)
        half = self._scast(0.5)

        # SYMMETRIC APPROACH
        if approach == 'symm':
            # create list to store convergence
            convergence = []
            # Take random orthonormal initial vectors.
            Q = utils.random_rot(comp, typecode)
            QOld = numx.zeros(numx.shape(Q),typecode)
            # This is the actual fixed-point iteration loop.
            for round in range(max_it + 1):
                if round == max_it:
                    raise mdp.NodeException,\
                          'No convergence after %d steps\n'%max_it
                # Symmetric orthogonalization. Q = Q * real(inv(Q' * Q)^(1/2));
                Q = mult(Q, utils.sqrtm(utils.inv(mult(t(Q), Q))))
                # Test for termination condition.Note that we consider opposite
                # directions here as well.
                convergence.append(1 - utils.amin(abs(\
                    utils.diag(mult(t(Q),QOld))),axis=0))
                if convergence[round] < limit:
                    if verbose: print 'Convergence after %d steps\n'%round
                    break

                QOld = Q
                # Show the progress...
                if verbose: print 'Step no. %d, convergence: %.3f'\
                   %(round+1, convergence[round])

                # First calculate the independent components (u_i's).
                # u_i = b_i' x = x' b_i. For all x:s simultaneously this is
                u = mult(t(X),Q)
                # non linearity
                if g == 'pow3':
                    Q = mult(X,u*u*u)/tlen - three*Q
                elif g == 'tanh':
                    tang = numx.tanh(fine_tanh * u)
                    temp = t(numx.sum(one-tang*tang))/tlen
                    Q = mult(X,tang) - temp * Q * fine_tanh
                elif g == 'gaus':
                    u2 = u*u
                    gauss =  u*numx.exp(-fine_gaus*u2*half)
                    dgauss = (one - fine_gaus*u2)*numx.exp(-fine_gaus*u2*half)
                    Q = (mult(X,gauss) - numx.sum(dgauss)*Q)/tlen
            self.convergence = numx.array(convergence)
            ret = convergence[-1]
         # DEFLATION APPROACH
        elif approach == 'defl':
            # adjust limit! 
            limit = 1 - limit*limit*0.5
            # create array to store convergence
            convergence = numx.zeros((comp,),typecode=typecode)
            Q = numx.zeros((comp,comp),typecode=typecode)
            round = 0
            nfail = 0
            while round < comp:
                # Take a random initial vector of lenght 1 and orthogonalize it
                # with respect to the other vectors.
                w  = self._refcast(numx_rand.random((comp, 1)) - .5)
                w -= mult(mult(Q,t(Q)),w)
                w /= utils.norm2(w)
                wOld = numx.zeros(numx.shape(w), typecode)
                # This is the actual fixed-point iteration loop.
                for i in range(max_it + 1):
                    if i == max_it:
                        round = round - 1
                        nfail += 1
                        if nfail > failures:
                            erstr = 'Too many failures to converge (%d).' % \
                                    nfail+ ' Giving up.'
                            raise mdp.NodeException, erstr 
                        break
                    # Project the vector into the space orthogonal to the space
                    # spanned by the earlier found basis vectors. Note that
                    # we can do the projection with matrix Q, since the zero
                    # entries do not contribute to the projection.
                    w -= mult(mult(Q,t(Q)),w)
                    w /= utils.norm2(w)
                    # Test for termination condition. Note that the algorithm
                    # has converged if the direction of w and wOld is the same.
                    conv = float(utils.squeeze(abs(numx.sum(w*wOld))))
                    if conv >= limit:
                        nfail = 0
                        convergence[round] = 1 - conv
                        # Calculate ICA filter.
                        Q[:,round] = w[:,0]
                        # Show the progress...
                        if verbose: print 'IC %d computed ( %d steps )'\
                           %(round+1, i+1)
                        break

                    wOld = w
                    # First calculate the independent components (u_i's) for
                    # this w.
                    # u_i = b_i' x = x' b_i. For all x:s simultaneously this is
                    u = mult(t(X),w)
                    #non linearity
                    if g == 'pow3':
                        w = mult(X,u*u*u)/tlen - three*w
                    elif g == 'tanh':
                        tang = numx.tanh(fine_tanh * u)
                        temp = t(numx.sum(one - tang*tang))*w
                        w = mult(X,tang) - fine_tanh/tlen*temp
                    elif g == 'gaus':
                        u2 = u*u
                        temp = numx.exp(-fine_gaus * u2*half)
                        gauss =  u *temp
                        dgauss = (one - fine_gaus *u2)*temp
                        w = (mult(X,gauss) - numx.sum(dgauss) * w) / tlen
                    # Normalize the new w.
                    w /= utils.norm2(w)
                round = round + 1
            # avoid roundoff errors
            dummy_cond = numx.logical_and(convergence<0, convergence>-1E-5)
            convergence = numx.where(dummy_cond, 0, convergence)
            self.convergence = self._refcast(numx.sqrt(convergence*2))
            ret = utils.amax(self.convergence)
        self.filters = Q
        return ret
