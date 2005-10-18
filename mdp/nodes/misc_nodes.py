import mdp
from mdp import numx, numx_linalg, utils, FiniteNode, NodeException

class OneDimensionalHitParade(object):
    """
    Class to produce hit-parades out of a one-dimensional time-series.
    """
    def __init__(self,n,d,real_typecode="d",integer_typecode="l"):
        """
        n - number of maxima and minima to remember
        d - minimum gap between two hits

        real_typecode is for sequence items
        integer_typecode is for sequence indices

        Note: be careful with typecodes!
        """
        self.n = int(n)
        self.d = int(d)
        self.iM = numx.zeros((n,),typecode=integer_typecode)
        self.im = numx.zeros((n,),typecode=integer_typecode)
        self.M = numx.array([-utils.inf]*n, typecode=real_typecode)
        self.m = numx.array([utils.inf]*n, typecode=real_typecode)
        self.lM = 0
        self.lm = 0

    def update(self,inp):
        """
        inp is the tuple (time-series, time-indices)
        """
        argmin = numx.argmin
        argmax = numx.argmax
        (x,ix) = inp
        rows = len(x)
        d = self.d
        M = self.M
        m = self.m
        iM = self.iM
        im = self.im
        lM = self.lM
        lm = self.lm
        for i in xrange(rows):
            k1 = argmin(M)
            k2 = argmax(m)
            if x[i] > M[k1]:
                if ix[i]-iM[lM] <= d and x[i] > M[lM]:
                    M[lM] = x[i]
                    iM[lM] = ix[i]
                elif ix[i]-iM[lM] > d:
                    M[k1] = x[i]
                    iM[k1] = ix[i]
                    lM = k1
            if x[i] < m[k2]:
                if ix[i]-im[lm] <= d and x[i] < m[lm]:
                    m[lm] = x[i]
                    im[lm] = ix[i]
                elif ix[i]-im[lm] > d:
                    m[k2] = x[i]
                    im[k2] = ix[i]
                    lm = k2
        self.M = M
        self.m = m
        self.iM = iM
        self.im = im
        self.lM = lM
        self.lm = lm

    def get_maxima(self):
        """
        return the tuple (maxima,time-indices)
        maxima are sorted largest-first
        """
        iM = self.iM
        M = self.M
        sort = numx.argsort(M)
        return numx.take(M,sort[::-1]),numx.take(iM,sort[::-1])
        
    def get_minima(self):
        """
        return the tuple (minima,time-indices)
        minima are sorted smallest-first
        """
        im = self.im
        m = self.m
        sort = numx.argsort(m)
        return numx.take(m,sort), numx.take(im,sort)

    
    
class HitParadeNode(FiniteNode):
    """HitParadeNode gets a multidimensional input signal and stores the first
    'n' local maxima and minima, which are separated by a minimum gap 'd'.
    This is called HitParade.

    Note: this node can be pickled with binary protocols only if
    all HitParade items are different from float 'inf', because
    of a bug in pickle [1]. If this is the case for you please use
    the pickle ASCII protocol '0'.

    This is an analysis node, i.e. the data is analyzed during training
    and the results are stored internally.

    References:
    [1] Pickle bug: [ 714733 ] cPickle fails to pickle inf
        https://sourceforge.net/tracker/?func=detail&atid=105470&aid=714733&group_id=5470

    """    

    
    def __init__(self, n, d, input_dim=None, typecode=None):
        """
        n - number of maxima and minima to store
        d - minimum gap between two maxima or two minima
        """
        super(HitParadeNode, self).__init__(input_dim, None, typecode)
        self.n = int(n)
        self.d = int(d)
        self.itype = 'l'
        self.hit = None
        self.tlen = 0

    def get_supported_typecodes(self):
        return ['i','l','f','d']

    def _train(self, x):
        hit = self.hit
        old_tlen = self.tlen
        if hit is None:
            hit = [OneDimensionalHitParade\
                   (self.n,self.d,self._typecode,self.itype)\
                   for c in range(self._input_dim)]
        tlen = old_tlen + x.shape[0]
        indices = numx.arange(old_tlen,tlen)
        for c in range(self._input_dim):
            hit[c].update((x[:,c],indices))
        self.hit = hit
        self.tlen = tlen

    def _stop_training(self):
        pass

    def copy(self, protocol = 0):
        """Return a deep copy of the node.
        Note: we use pickle protocol '0' since
        binary protocols can not pickle
        float 'inf'."""
        return super(HitParadeNode, self).copy(protocol)
    
    def get_maxima(self):
        """
        Return the tuple (maxima, indices).
        Maxima are sorted largest-first.
        """
        cols = self._input_dim
        n = self.n
        hit = self.hit
        iM = numx.zeros((n,cols),typecode=self.itype)
        M = numx.ones((n,cols),typecode=self._typecode)
        for c in range(cols):
            M[:,c],iM[:,c] = hit[c].get_maxima()
        return M,iM
    
    def get_minima(self):
        """
        Return the tuple (minima, indices).
        Minima are sorted smallest-first.
        """
        cols = self._input_dim
        n = self.n
        hit = self.hit
        im = numx.zeros((n,cols), typecode=self.itype)
        m = numx.ones((n,cols), typecode=self._typecode)
        for c in range(cols):
            m[:,c],im[:,c] = hit[c].get_minima()
        return m,im

class TimeFramesNode(FiniteNode):
    """TimeFramesNode receives a multidimensional input signal and copies on
    the space dimensions delayed version of the same signal. Example:

    If time_frames=3 and gap=2: 
    
    [ X(1) Y(1)        [ X(1) Y(1) X(3) Y(3) X(5) Y(5)
      X(2) Y(2)          X(2) Y(2) X(4) Y(4) X(6) Y(6)]  
      X(3) Y(3)   -->      
      X(4) Y(4)
      X(5) Y(5)
      X(6) Y(6)]

    """
    
    def __init__(self, time_frames, gap=1, input_dim=None, typecode=None):     
        super(TimeFramesNode, self).__init__(input_dim, None, typecode)
        self.time_frames = time_frames
        self.gap = gap

    def is_trainable(self):
        return 0

    def is_invertible(self):
        return 0
        
    def _set_default_outputdim(self, nvariables):
        self._output_dim = nvariables*self.time_frames
            
    def _execute(self, x):
        gap = self.gap
        tf = x.shape[0]- (self.time_frames-1)*gap
        rows = self._input_dim
        cols = self._output_dim
        y = numx.zeros((tf,cols),typecode=self._typecode)
        for frame in range(self.time_frames):
            y[:,frame*rows:(frame+1)*rows] = x[gap*frame:gap*frame+tf,:]
        return y

    def pseudo_inverse(self, y):
        """This function returns a pseudo-inverse of the execute frame.
        y == execute(x) only if y belongs to the domain of execute and
        has been computed with a sufficently large x."""
        
        self._if_training_stop_training()

        # set the output dimension if necessary
        if not self._output_dim:
            # if the input_dim is not defined, raise an exception
            if not self._input_dim:
                errstr = "Number of input dimensions undefined. Inversion"+\
                         "not possible."
                raise NodeException, errstr
            self._set_default_outputdim(self._input_dim)
        
        # control the dimension of y
        self._check_output(y)
        # cast
        y = self._refcast(y)
        
        gap = self.gap
        exp_length = y.shape[0]
        cols = self._input_dim
        rest = (self.time_frames-1)*gap
        rows = exp_length + rest
        x = numx.zeros((rows,cols),typecode=self._typecode)
        x[:exp_length,:] = y[:,:cols]
        count = 1
        # Note that if gap > 1 some of the last rows will be filled with zeros!
        block_sz = min(gap, exp_length)
        for row in range(max(exp_length,gap),rows,gap):
            x[row:row+block_sz,:] = y[-block_sz:,count*cols:(count+1)*cols]
            count += 1
        return x


class EtaComputerNode(FiniteNode):
    """Node to compute the eta values of the normalized training data.

    The delta value of a signal is a measure of its temporal
    variation, and is defined as the mean of the derivative squared,
    i.e. delta(x) = mean(dx/dt(t)^2).  delta(x) is zero if
    x is a constant signal, and increases if the temporal variation
    of the signal is bigger.

    The eta value is a more intuitive measure of temporal variation,
    defined as
       eta(x) = T/(2*pi) * sqrt(delta(x))
    If x is a signal of length T which consists of a sine function
    that accomplishes exactly N oscillations, then eta(x)=N.

    EtaComputerNode normalizes the training data to have unit
    variance, such that it is possible to compare the temporal
    variation of two signals independently from their scaling.

    Reference: Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis:
    Unsupervised Learning of Invariances, Neural Computation,
    14(4):715-770 (2002).

    Important: if a data chunk is tlen data points long, this node is
    going to consider only the first tlen-1 points together with their
    derivatives. This means in particular that the variance of the
    signal is not computed on all data points. This behavior is
    compatible with that of SFANode.

    This is an analysis node, i.e. the data is analyzed during training
    and the results are stored internally.
    """
    
    def __init__(self, input_dim=None, typecode=None):
        super(EtaComputerNode, self).__init__(input_dim, None, typecode)
        self._initialized = 0

    def get_supported_typecodes(self):
        return ['f','d']

    def _init_internals(self):
        input_dim = self._input_dim
        self._mean = numx.zeros((input_dim,), typecode='d')
        self._var = numx.zeros((input_dim,), typecode='d')
        self._tlen = 0
        self._diff2 = numx.zeros((input_dim,), typecode='d')
        self._initialized = 1

    def _train(self, data):
        # ?? refcast automatico
        if not self._initialized: self._init_internals()
        #
        rdata = data[:-1]
        self._mean += numx.sum(rdata, axis=0)
        self._var += numx.sum(rdata*rdata, axis=0)
        self._tlen += rdata.shape[0]
        self._diff2 += numx.sum(utils.timediff(data)**2)

    def _stop_training(self):
        var_tlen = self._tlen-1
        var = (self._var/var_tlen) - (self._mean/self._tlen)**2
        delta = (self._diff2/self._tlen)/var
        self._eta = numx.sqrt(delta)/(2*numx.pi)

    def get_eta(self, t=1):
        """Return the eta values of the data received during the training
        phase. If the training phase has not been completed yet, call
        stop_training."""
        if self.is_training(): self.stop_training()
        return self._refcast(self._eta*t)


class NoiseNode(FiniteNode):
    """Node to add noise to input data.

    Original idea by Mathias Franzius.
    """
    def __init__(self, noise_func = utils.normal, noise_args = (0,1),
                 noise_type = 'additive', input_dim = None, typecode = None):
        """
        Add noise to input signals.
        
        - 'noise_func' must take a 'shape' keyword argument and return
          a random array of that size. Default is normal noise.
          
        - 'noise_args' is a tuple of additional arguments for the noise_func.
          Default is (0,1) for (mean, standard deviation) of the normal
          distribution.

        - 'noise_type' is either 'additive' or 'multiplicative':
            'additive' returns x + noise
            'multiplicative' returns x * (1 + noise)
          Default is 'additive'.
        """
        super(NoiseNode, self).__init__(input_dim = input_dim,
                                        typecode = typecode)
        self.noise_func = noise_func
        self.noise_args = noise_args
        valid_noise_types = ['additive', 'multiplicative']
        if noise_type not in valid_noise_types:
            err_str = '%s is not a valid noise type' % str(noise_type)
            raise NodeException, err_str
        else:
            self.noise_type = noise_type
            
    def is_trainable(self):
        return 0

    def is_invertible(self):
        return 0

    def _execute(self, x):
        noise_mat = self._refcast(self.noise_func(*self.noise_args,
                                                  **{'shape': x.shape}))
        if self.noise_type == 'additive':
            return x+noise_mat
        elif self.noise_type == 'multiplicative':
            return x*(self._scast(1)+noise_mat)


class GaussianClassifierNode(FiniteNode):
    def __init__(self, input_dim=None, typecode=None):
        """This node performs a supervised Gaussian classification.

        Given a set of labelled data, this node fits a gaussian distribution
        to each class. Note that it is written as an analysis node (i.e., the
        execute function is the identity function). To perform classification,
        use the 'classify' method. If instead you need the posterior porbability
        of the classes given the data use the 'class_probabilities' method.
        """
        super(GaussianClassifierNode, self).__init__(input_dim, None, typecode)
        self.cov_objs = {}

    def is_invertible(self):
        return False

    def _update_covs(self, x, lbl):
        if not self.cov_objs.has_key(lbl):
            self.cov_objs[lbl] = \
                mdp.nodes.lcov.CovarianceMatrix(typecode = self._typecode)
        self.cov_objs[lbl].update(x)

    def _train(self, x, cl):
        """'cl' can be a list of labels (one for each data point) or
        a single label, in which case all input data is assigned to
        the same class."""
        # if cl is a number, all x's belong to the same class
        if type(cl) is int:
            self._update_covs(x, cl)
        else:
            # get all classes from cl
            for lbl in  utils.uniq(cl):
                x_lbl = numx.compress(cl==lbl, x, axis=0)
                self._update_covs(x_lbl, lbl)

    def _stop_training(self):
        self.labels = self.cov_objs.keys()
        self.labels.sort()

        # we are going to store the inverse of the covariance matrices
        # since only those are useful to compute the probabilities
        self.inv_covs = []
        self.means = []
        self.p = []
        # this list contains the square root of the determinant of the
        # corresponding covariance matrix
        self._sqrt_def_covs = []
        nitems = 0
        
        for lbl in self.labels:
            cov, mean, p = self.cov_objs[lbl].fix()
            nitems += p
            self._sqrt_def_covs.append(numx.sqrt(numx_linalg.det(cov)))
            self.means.append(mean)
            self.p.append(p)
            self.inv_covs.append(numx_linalg.inv(cov))

        for i in range(len(self.p)):
            self.p[i] /= self._scast(nitems)

        del self.cov_objs

    # ?? if the distribution objects of the scipy.stats module were also
    # in Numeric and numarray we could use them
    def _gaussian_prob(self, x, lbl_idx):
        """Return the probability of the data points x with respect to a gaussian.
        x: input data, S: covariance matrix, mn: mean"""
        x = self._refcast(x)

        dim = self._input_dim
        sqrt_detS = self._sqrt_def_covs[lbl_idx]
        invS = self.inv_covs[lbl_idx]
        # subtract the mean
        x_mn = x - self.means[lbl_idx][numx.NewAxis,:]
        # exponent
        exponent = self._scast(-0.5) * \
                   numx.sum(utils.mult(x_mn, invS)*x_mn, axis=1)
        # constant
        constant = self._scast((2.*numx.pi)**-(dim/2.)/sqrt_detS)
        # probability
        return constant * numx.exp(exponent)

    def class_probabilities(self, x):
        """Return the probability of each class given the input."""
        self._pre_execution_checks(x)

        # compute the probability for each class
        tmp_prob = numx.zeros((x.shape[0], len(self.labels)),
                              typecode=self._typecode)
        for i in range(len(self.labels)):
            tmp_prob[:,i] = self._gaussian_prob(x, i)
            tmp_prob[:,i] *= self.p[i]
            
        # normalize to probability 1
        # (not necessary, but sometimes useful)
        tmp_tot = numx.sum(tmp_prob, axis=1)
        tmp_tot = tmp_tot[:, numx.NewAxis]
        return tmp_prob/tmp_tot
        
    def classify(self, x):
        """Classify the input data using Maximum A-Posteriori."""
        self._pre_execution_checks(x)

        class_prob = self.class_probabilities(x)
        winner = numx.argmax(class_prob)
        return [self.labels[winner[i]] for i in range(len(winner))]
