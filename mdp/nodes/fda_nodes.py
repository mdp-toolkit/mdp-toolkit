import mdp

numx = mdp.numx

class FDANode(mdp.Node):
    """FDANode is a supervised node that performs a (generalized)
    Fisher Discriminant Analysis of its input. It is a supervised node
    that implements FDA using a generalized eigenvalue approach.

    FDANode has two training phases and is supervised so make sure to
    pay attention to the following points when training it:
    
    - call the train function passing *two* arguments: the input data
      and the labels (see the doc string of the train method for details)
      
    - if you are training the node by hand call the train function twice
    
    - if you are training the node using a flow (recommended), the
      only argument of Flow.train must be a list of (data_point,
      label) tuples or an iterator returning lists of such tuples,
      *not* a generator.  The Flow.train function can be called just
      once as usual, since it takes care of "rewinding" the iterator
      to perform the second training step.
    
    More information on Fisher Discriminant Analysis can be found for
    example in C. Bishop, Neural Networks for Pattern Recognition,
    Oxford Press, pp. 105-112 ."""
    
    def _get_train_seq(self):
        return [(self._train_means, self._stop_means),
                (self._train_fda, self._stop_fda)]
    
    def __init__(self, input_dim=None, output_dim=None, typecode=None):
        super(FDANode, self).__init__(input_dim, output_dim, typecode)
        self.S_W = None
        self.allcov = mdp.nodes.lcov.CovarianceMatrix(typecode = self.typecode)
        self.means = {}
        self.tlens = {}
        self._SW_init = 0

    def _check_train_args(self, x, cl):
        if not isinstance(cl, int) and len(cl) != x.shape[0]:
            msg = "The number of labels should be equal to the number of " +\
                  "datapoints (%d != %d)" % (len(cl), x.shape[0])
            raise mdp.TrainingException, msg

    # Training step 1: compute mean and number of element in each class

    def _update_means(self, x, lbl):
        if not self.means.has_key(lbl):
            self.means[lbl] = numx.zeros((1, self.input_dim),
                                         dtype=self.typecode)
            self.tlens[lbl] = 0
        self.means[lbl] += numx.sum(x, 0)
        self.tlens[lbl] += x.shape[0]
    
    def _train_means(self, x, cl):
        if isinstance(cl, int):
            self._update_means(x, cl)
        else:
            for lbl in mdp.utils.uniq(cl):
                # group for class
                x_lbl = numx.compress(cl==lbl, x, axis=0)
                self._update_means(x_lbl, lbl)

    def _stop_means(self):
        for lbl in self.means.keys():
            self.means[lbl] /= self._scast(self.tlens[lbl])

    # Training step 2: compute the overall and within-class covariance
    # matrices and solve the FDA problem

    def _update_SW(self, x, lbl):
        x = x - self.means[lbl]
        # update S_W
        self.S_W += mdp.utils.mult(numx.transpose(x), x)
 
    def _train_fda(self, x, cl):
        #if self.S_W == None:
        if self._SW_init == 0:
            self._SW_init = 1
            self.S_W = numx.zeros((self.input_dim, self.input_dim),
                                  dtype=self.typecode)

        # update the covariance matrix of all classes
        self.allcov.update(x)

        # if cl is a number, all x's belong to the same class
        if isinstance(cl, int):
            self._update_SW(x, cl)
        else:
            # get all classes from cl
            for lbl in mdp.utils.uniq(cl):
                # group for class
                x_lbl = numx.compress(cl==lbl, x, axis=0)
                self._update_SW(x_lbl, lbl)

    def _stop_fda(self):
        S_T, self.avg, tlen = self.allcov.fix()
        del self.allcov
        S_W = self.S_W
        del self.S_W
       
        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        if self.output_dim is None:
            rng = None
            self.output_dim = self.input_dim
        else:
            rng = (1, self.output_dim)
            
        d, self.v = mdp.utils.symeig(S_W, S_T, range=rng, overwrite = 1)

    def train(self, x, cl):
        """Update the internal structures according to the input data 'x'.
        
        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list of labels (one for each data point) or
              a single label, in which case all input data is assigned to
              the same class.
        """
        super(FDANode, self).train(x, cl)

    def _execute(self, x, range=None):
        if range:
            if isinstance(range, (list, tuple)):
                v = self.v[:,range[0]:range[1]]
            else:
                v = self.v[:,0:range]
        else:
            v = self.v

        return mdp.utils.mult(x-self.avg, v)

    def _inverse(self, y):
        return mdp.utils.mult(y, mdp.utils.pinv(self.v))+self.avg
