import mdp

numx = mdp.numx

class FDANode(mdp.Node):
    """Perform a (generalized) Fisher Discriminant Analysis of its
    input. It is a supervised node that implements FDA using a
    generalized eigenvalue approach.

    FDANode has two training phases and is supervised so make sure to
    pay attention to the following points when you train it:

    - call the 'train' function with *two* arguments: the input data
      and the labels (see the doc string of the train method for details)

    - if you are training the node by hand, call the train function twice

    - if you are training the node using a flow (recommended), the
      only argument to Flow.train must be a list of (data_point,
      label) tuples or an iterator returning lists of such tuples,
      *not* a generator.  The Flow.train function can be called just
      once as usual, since it takes care of "rewinding" the iterator
      to perform the second training step.

    More information on Fisher Discriminant Analysis can be found for
    example in C. Bishop, Neural Networks for Pattern Recognition,
    Oxford Press, pp. 105-112.

    Internal variables of interest:
    self.avg -- Mean of the input data (available after training)
    self.v -- Transposed of the projection matrix, so that
              output = dot(input-self.avg, self.v)
              (available after training)
    """

    def _get_train_seq(self):
        return [(self._train_means, self._stop_means),
                (self._train_fda, self._stop_fda)]

    def __init__(self, input_dim=None, output_dim=None, dtype=None):
        super(FDANode, self).__init__(input_dim, output_dim, dtype)
        # mean in-class covariance matrix times number of data points
        # is deleted after training
        self._S_W = None
        # covariance matrix of the full data distribution
        self._allcov = mdp.utils.CovarianceMatrix(dtype=self.dtype)
        self.means = {}  # maps class labels to the class means
        self.tlens = {}  # maps class labels to number of training points
        self.v = None  # transposed of the projection matrix
        self.avg = None  # mean of the input data

    def _check_train_args(self, x, cl):
        if (isinstance(cl, (list, tuple, numx.ndarray)) and
            len(cl) != x.shape[0]):
            msg = ("The number of labels should be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)

    # Training step 1: compute mean and number of elements in each class

    def _train_means(self, x, cl):
        """Gather data to compute the means and number of elements."""
        if isinstance(cl, (list, tuple, numx.ndarray)):
            labels_ = numx.asarray(cl)
            for lbl in set(labels_):
                # group for class
                x_lbl = numx.compress(labels_==lbl, x, axis=0)
                self._update_means(x_lbl, lbl)
        else:
            self._update_means(x, cl)

    def _stop_means(self):
        """Calculate the class means."""
        for lbl in self.means:
            self.means[lbl] /= self.tlens[lbl]

    def _update_means(self, x, lbl):
        """Update the internal variables that store the data for the means.

        x -- Data points from a single class.
        lbl -- The label for that class.
        """
        if lbl not in self.means:
            self.means[lbl] = numx.zeros((1, self.input_dim), dtype=self.dtype)
            self.tlens[lbl] = 0
        self.means[lbl] += x.sum(axis=0)
        self.tlens[lbl] += x.shape[0]

    # Training step 2: compute the overall and within-class covariance
    # matrices and solve the FDA problem

    def _train_fda(self, x, cl):
        """Gather data for the overall and within-class covariance"""
        if self._S_W is None:
            self._S_W = numx.zeros((self.input_dim, self.input_dim),
                                   dtype=self.dtype)
        # update the covariance matrix of all classes
        self._allcov.update(x)
        # if cl is a number, all x's belong to the same class
        if isinstance(cl, (list, tuple, numx.ndarray)):
            labels_ = numx.asarray(cl)
            # get all classes from cl
            for lbl in set(labels_):
                # group for class
                x_lbl = numx.compress(labels_==lbl, x, axis=0)
                self._update_SW(x_lbl, lbl)
        else:
            self._update_SW(x, cl)

    def _stop_fda(self):
        """Solve the eigenvalue problem for the total covariance."""
        S_T, self.avg, _ = self._allcov.fix()
        del self._allcov
        S_W = self._S_W
        del self._S_W
        # solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        if self.output_dim is None:
            rng = None
            self.output_dim = self.input_dim
        else:
            rng = (1, self.output_dim)
        self.v = mdp.utils.symeig(S_W, S_T, range=rng, overwrite = 1)[1]

    def _update_SW(self, x, lbl):
        """Update the covariance matrix of the class means.

        x -- Data points from a single class.
        lbl -- The label for that class.
        """
        x = x - self.means[lbl]
        self._S_W += mdp.utils.mult(x.T, x)

    # Overwrite the standard methods

    def _train(self, x, cl):
        """Update the internal structures according to the input data 'x'.

        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """
        super(FDANode, self).train(x, cl)

    def _execute(self, x, range=None):
        """Compute the output of the FDA projection.

        if 'range' is a number, then use the first 'range' functions.
        if 'range' is the interval=(i,j), then use all functions
        between i and j.
        """
        if range:
            if isinstance(range, (list, tuple)):
                v = self.v[:, range[0]:range[1]]
            else:
                v = self.v[:, 0:range]
        else:
            v = self.v
        return mdp.utils.mult(x-self.avg, v)

    def _inverse(self, y):
        return mdp.utils.mult(y, mdp.utils.pinv(self.v))+self.avg
