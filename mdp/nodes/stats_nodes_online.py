import mdp


class OnlineCenteringNode(mdp.PreserveDimOnlineNode):
    """OnlineCenteringNode centers the input data, that is, subtracts the arithmetic mean (average) from the
    input data. This is an online learnable node.

    The node's train method updates the average (avg) according to the update rule:

        avg <- (1 / n) * x + (1-1/n) * avg, where n is the total number of samples observed while training.

    The node's execute method subtracts the updated average from the input and returns it.

    This node also supports centering via an exponentially weighted moving average that resembles a leaky
    integrator:

        avg <- alpha * x + (1-alpha) * avg, where alpha = 2. / (avg_n + 1).

    avg_n intuitively denotes a "window size". For a large avg_n, 'avg_n'-samples represent about 86% of
    the total weight.

    **Internal variables of interest (stored in cache)**

      ``self.avg``
          The updated average of the input data

    """

    def __init__(self, avg_n=None, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """
        avg_n - If set to None (default), the node updates a simple moving average. If set to a positive integer,
        the node updates an exponentially weighted moving average.
        """
        super(OnlineCenteringNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype,
                                                  numx_rng=numx_rng)
        self.avg_n = avg_n
        self.avg = None
        self._cache = {'avg': None}

    def _check_params(self, x):
        if self.avg is None:
            self.avg = mdp.numx.zeros(x.shape[1], dtype=self.dtype)

    def _get_supported_training_types(self):
        return ['incremental']

    def _train(self, x):
        """updates the average parameter"""
        if self.avg_n is None:
            alpha = 1.0 / (self.get_current_train_iteration() + 1.)
        else:
            alpha = 2.0 / (self.avg_n + 1) if self.get_current_train_iteration() > 0 else 1.0

        self.avg = (1 - alpha) * self.avg + alpha * x
        self._cache['avg'] = self.avg

    def _execute(self, x):
        """returns a centered input"""
        if self.get_current_train_iteration() <= 1:
            return x
        else:
            return x - self.avg

    def _inverse(self, x):
        """returns the non-centered original data"""
        if self.get_current_train_iteration() <= 1:
            return x
        else:
            return x + self.avg

    def get_average(self):
        """returns the updated average"""
        return self.avg

    def __repr__(self):
        # print all args
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" % self.dtype.name
        numx_rng = "numx_rng=%s" % str(self.numx_rng)
        avg_n = "avg_n=%s" % self.avg_n
        args = ', '.join((inp, out, typ, numx_rng, avg_n))
        return name + '(' + args + ')'


class OnlineTimeDiffNode(mdp.PreserveDimOnlineNode):
    """Compute the discrete time derivative of the input using backward difference approximation:

        dx(n) = x(n) - x(n-1), where n is the total number of input samples observed during training.

    This is an online learnable node that uses a buffer to store the previous input sample = x(n-1). The node's train
    method updates the buffer. The node's execute method returns the time difference using the stored buffer
    as its previous input sample x(n-1).

    Few example usages:

    If the training and execute methods are called sample by sample incrementally:
        train(x[1]), y[1]=execute(x[1]), train(x[2]), y[2]=execute(x[2]), ...,
    then
        y[1] = x[1]
        y[2] = x[2] - x[1]
        y[3] = x[3] - x[2]
        ...

    If training and execute methods are called block by block:
        train([x[1], x[2], x[3]]), [y[3], y[4], y[5]] = execute([x[3], x[4], x[5]])
    then
        y[3] = x[3] - x[2]
        y[4] = x[4] - x[3]
        y[5] = x[5] - x[4]

    Note that the stored buffer is still = x[2]. Only train() method changes the state of the node.
    execute's input data is always assumed to start at get_current_train_iteration() time step.

    This node supports both "incremental" and "batch" training types.
    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        super(OnlineTimeDiffNode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.x_prev = None
        self.x_cur = None

    def _check_params(self, x):
        if self.x_prev is None:
            self.x_prev = mdp.numx.zeros(x.shape, dtype=self.dtype)
            self.x_cur = mdp.numx.zeros(x.shape, dtype=self.dtype)

    def _train(self, x):
        """Update the buffer"""
        self.x_prev = self.x_cur
        self.x_cur = x[-1:]

    @staticmethod
    def is_invertible():
        return False

    def _execute(self, x):
        """returns the time difference"""
        x = mdp.numx.vstack((self.x_prev, x))
        return x[1:] - x[:-1]
