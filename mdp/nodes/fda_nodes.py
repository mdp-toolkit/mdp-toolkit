import mdp

numx = mdp.numx

class FDANode(mdp.FiniteNode):
    
    def get_train_seq(self):
        return [(self.train_means, self.stop_means),
                (self.train_fda, self.stop_fda)]
    
    def __init__(self, input_dim=None, output_dim=None, typecode=None):
        """Perform Fisher Discriminant Analysis.

        This is a supervised node that implements FDA using a
        generalized eigenvalue approach.
        """
        super(FDANode, self).__init__(input_dim, output_dim, typecode)
        self.S_W = None
        self.allcov = mdp.nodes.lcov.CovarianceMatrix(typecode = self._typecode)
        self.means = {}
        self.tlens = {}
        self._SW_init = 0

    def _check_train_args(self, x, cl):
        if type(cl) is not int and len(cl)!=x.shape[0]:
            msg = "The number of labels should be equal to the number of " +\
                  "datapoints (%d != %d)" % (len(cl), x.shape[0])
            raise mdp.TrainingException, msg

    # Training step 1: compute mean and number of element in each class

    def _update_means(self, x, lbl):
        if not self.means.has_key(lbl):
            self.means[lbl] = numx.zeros((1, self._input_dim), typecode=self._typecode)
            self.tlens[lbl] = 0
        self.means[lbl] += numx.sum(x, 0)
        self.tlens[lbl] += x.shape[0]
    
    def train_means(self, x, cl):
        """'cl' can be a list of labels (one for each data point) or
        a single label, in which case all input data is assigned to
        the same class."""

        if type(cl) is int:
            self._update_means(x, cl)
        else:
            for lbl in mdp.utils.uniq(cl):
                # group for class
                x_lbl = numx.compress(cl==lbl, x, axis=0)
                self._update_means(x_lbl, lbl)

    def stop_means(self):
        for lbl in self.means.keys():
            self.means[lbl] /= self._scast(self.tlens[lbl])

    # Training step 2: compute the overall and within-class covariance
    # matrices and solve the FDA problem

    def _update_SW(self, x, lbl):
        x = x - self.means[lbl]
        # update S_W
        self.S_W += mdp.utils.mult(numx.transpose(x), x)
 
    def train_fda(self, x, cl):
        #if self.S_W == None:
        if self._SW_init == 0:
            self._SW_init = 1
            self.S_W = numx.zeros((self._input_dim, self._input_dim),
                                  typecode=self._typecode)

        # update the covariance matrix of all classes
        self.allcov.update(x)

        # if cl is a number, all x's belong to the same class
        if type(cl) is int:
            self._update_SW(x, cl)
        else:
            # get all classes from cl
            for lbl in mdp.utils.uniq(cl):
                # group for class
                x_lbl = numx.compress(cl==lbl, x, axis=0)
                self._update_SW(x_lbl, lbl)

    def stop_fda(self):
        S_T, self.avg, tlen = self.allcov.fix()
        del self.allcov
        S_W = self.S_W
        del self.S_W
       
        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        if self._output_dim>=1:
            rng = (1, self._output_dim)
        else:
            rng = None

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
            if type(range) is types.ListType or type(range) is types.TupleType:
                v = self.v[:,range[0]:range[1]]
            else:
                v = self.v[:,0:range]
        else:
            v = self.v

        return mdp.utils.mult(x-self.avg, v)

    def _inverse(self, y):
        return mdp.utils.mult(y, mdp.utils.pinv(self.v))+self.avg
