from __future__ import division
from future import standard_library

standard_library.install_aliases()
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import inspect as _inspect

__docformat__ = "restructuredtext en"

import mdp
from mdp import numx, utils, Node, NodeException, PreserveDimNode, config

import pickle as pickle
import pickle as real_pickle
import itertools


class IdentityNode(PreserveDimNode):
    """Execute returns the input data and the node is not trainable.

    This node can be instantiated and is for example useful in
    complex network layouts.
    """

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return (mdp.utils.get_dtypes('AllFloat') +
                mdp.utils.get_dtypes('AllInteger') +
                mdp.utils.get_dtypes('Character'))

    @staticmethod
    def is_trainable():
        return False


class OneDimensionalHitParade(object):
    """
    Class to produce hit-parades (i.e., a list of the largest
    and smallest values) out of a one-dimensional time-series.
    """

    def __init__(self, n, d, real_dtype="d", integer_dtype="l"):
        """
        Input arguments:
        n -- Number of maxima and minima to remember
        d -- Minimum gap between two hits

        real_dtype -- dtype of sequence items
        integer_dtype -- dtype of sequence indices
        Note: be careful with dtypes!
        """
        self.n = int(n)
        self.d = int(d)
        self.iM = numx.zeros((n,), dtype=integer_dtype)
        self.im = numx.zeros((n,), dtype=integer_dtype)

        real_dtype = numx.dtype(real_dtype)
        if real_dtype in mdp.utils.get_dtypes('AllInteger'):
            max_num = numx.iinfo(real_dtype).max
            min_num = numx.iinfo(real_dtype).min
        else:
            max_num = numx.finfo(real_dtype).max
            min_num = numx.finfo(real_dtype).min
        self.M = numx.array([min_num] * n, dtype=real_dtype)
        self.m = numx.array([max_num] * n, dtype=real_dtype)

        self.lM = 0
        self.lm = 0

    def update(self, inp):
        """
        Input arguments:
        inp -- tuple (time-series, time-indices)
        """
        (x, ix) = inp
        rows = len(x)
        d = self.d
        M = self.M
        m = self.m
        iM = self.iM
        im = self.im
        lM = self.lM
        lm = self.lm
        for i in range(rows):
            k1 = M.argmin()
            k2 = m.argmax()
            if x[i] > M[k1]:
                if ix[i] - iM[lM] <= d and x[i] > M[lM]:
                    M[lM] = x[i]
                    iM[lM] = ix[i]
                elif ix[i] - iM[lM] > d:
                    M[k1] = x[i]
                    iM[k1] = ix[i]
                    lM = k1
            if x[i] < m[k2]:
                if ix[i] - im[lm] <= d and x[i] < m[lm]:
                    m[lm] = x[i]
                    im[lm] = ix[i]
                elif ix[i] - im[lm] > d:
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
        Return the tuple (maxima, time-indices).
        Maxima are sorted in descending order.
        """
        iM = self.iM
        M = self.M
        sort = M.argsort()
        return M[sort[::-1]], iM[sort[::-1]]

    def get_minima(self):
        """
        Return the tuple (minima, time-indices).
        Minima are sorted in ascending order.
        """
        im = self.im
        m = self.m
        sort = m.argsort()
        return m[sort], im[sort]


class HitParadeNode(PreserveDimNode):
    """Collect the first ``n`` local maxima and minima of the training signal
    which are separated by a minimum gap ``d``.

    This is an analysis node, i.e. the data is analyzed during training
    and the results are stored internally. Use the
    ``get_maxima`` and ``get_minima`` methods to access them.
    """

    def __init__(self, n, d=1, input_dim=None, output_dim=None, dtype=None):
        """
        Input arguments:
        n -- Number of maxima and minima to store
        d -- Minimum gap between two maxima or two minima
        """
        super(HitParadeNode, self).__init__(input_dim=input_dim,
                                            output_dim=output_dim,
                                            dtype=dtype)
        self.n = int(n)
        self.d = int(d)
        self.itype = 'int64'
        self.hit = None
        self.tlen = 0

    def _set_input_dim(self, n):
        self._input_dim = n
        self.output_dim = n

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return (mdp.utils.get_dtypes('Float') +
                mdp.utils.get_dtypes('AllInteger'))

    def _train(self, x):
        hit = self.hit
        old_tlen = self.tlen
        if hit is None:
            hit = [OneDimensionalHitParade(self.n, self.d, self.dtype,
                                           self.itype)
                   for c in range(self.input_dim)]
        tlen = old_tlen + x.shape[0]
        indices = numx.arange(old_tlen, tlen)
        for c in range(self.input_dim):
            hit[c].update((x[:, c], indices))
        self.hit = hit
        self.tlen = tlen

    def get_maxima(self):
        """
        Return the tuple (maxima, indices).
        Maxima are sorted in descending order.

        If the training phase has not been completed yet, call
        stop_training.
        """
        self._if_training_stop_training()
        cols = self.input_dim
        n = self.n
        hit = self.hit
        iM = numx.zeros((n, cols), dtype=self.itype)
        M = numx.ones((n, cols), dtype=self.dtype)
        for c in range(cols):
            M[:, c], iM[:, c] = hit[c].get_maxima()
        return M, iM

    def get_minima(self):
        """
        Return the tuple (minima, indices).
        Minima are sorted in ascending order.

        If the training phase has not been completed yet, call
        stop_training.
        """
        self._if_training_stop_training()
        cols = self.input_dim
        n = self.n
        hit = self.hit
        im = numx.zeros((n, cols), dtype=self.itype)
        m = numx.ones((n, cols), dtype=self.dtype)
        for c in range(cols):
            m[:, c], im[:, c] = hit[c].get_minima()
        return m, im


class TimeFramesNode(Node):
    """Copy delayed version of the input signal on the space dimensions.

    For example, for ``time_frames=3`` and ``gap=2``::

      [ X(1) Y(1)        [ X(1) Y(1) X(3) Y(3) X(5) Y(5)
        X(2) Y(2)          X(2) Y(2) X(4) Y(4) X(6) Y(6)
        X(3) Y(3)   -->    X(3) Y(3) X(5) Y(5) X(7) Y(7)
        X(4) Y(4)          X(4) Y(4) X(6) Y(6) X(8) Y(8)
        X(5) Y(5)          ...  ...  ...  ...  ...  ... ]
        X(6) Y(6)
        X(7) Y(7)
        X(8) Y(8)
        ...  ...  ]

    It is not always possible to invert this transformation (the
    transformation is not surjective. However, the ``pseudo_inverse``
    method does the correct thing when it is indeed possible.
    """

    def __init__(self, time_frames, gap=1,
                 input_dim=None, dtype=None):
        """
        Input arguments:
        time_frames -- Number of delayed copies
        gap -- Time delay between the copies
        """
        self.time_frames = time_frames
        super(TimeFramesNode, self).__init__(input_dim=input_dim,
                                             output_dim=None,
                                             dtype=dtype)
        self.gap = gap

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return (mdp.utils.get_dtypes('AllFloat') +
                mdp.utils.get_dtypes('AllInteger') +
                mdp.utils.get_dtypes('Character'))

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n * self.time_frames

    def _set_output_dim(self, n):
        msg = 'Output dim can not be explicitly set!'
        raise NodeException(msg)

    def _execute(self, x):
        gap = self.gap
        tf = x.shape[0] - (self.time_frames - 1) * gap
        rows = self.input_dim
        cols = self.output_dim
        y = numx.zeros((tf, cols), dtype=self.dtype)
        for frame in range(self.time_frames):
            y[:, frame * rows:(frame + 1) * rows] = x[gap * frame:gap * frame + tf, :]
        return y

    def pseudo_inverse(self, y):
        """This function returns a pseudo-inverse of the execute frame.
        y == execute(x) only if y belongs to the domain of execute and
        has been computed with a sufficently large x.
        If gap > 1 some of the last rows will be filled with zeros.
        """

        self._if_training_stop_training()

        # set the output dimension if necessary
        if not self.output_dim:
            # if the input_dim is not defined, raise an exception
            if not self.input_dim:
                errstr = ("Number of input dimensions undefined. Inversion"
                          "not possible.")
                raise NodeException(errstr)
            self.outputdim = self.input_dim

        # control the dimension of y
        self._check_output(y)
        # cast
        y = self._refcast(y)

        gap = self.gap
        exp_length = y.shape[0]
        cols = self.input_dim
        rest = (self.time_frames - 1) * gap
        rows = exp_length + rest
        x = numx.zeros((rows, cols), dtype=self.dtype)
        x[:exp_length, :] = y[:, :cols]
        count = 1
        # Note that if gap > 1 some of the last rows will be filled with zeros!
        block_sz = min(gap, exp_length)
        for row in range(max(exp_length, gap), rows, gap):
            x[row:row + block_sz, :] = y[-block_sz:, count * cols:(count + 1) * cols]
            count += 1
        return x


class TimeDelayNode(TimeFramesNode):
    """
    Copy delayed version of the input signal on the space dimensions.

    For example, for ``time_frames=3`` and ``gap=2``::

      [ X(1) Y(1)        [ X(1) Y(1)   0    0    0    0
        X(2) Y(2)          X(2) Y(2)   0    0    0    0
        X(3) Y(3)   -->    X(3) Y(3) X(1) Y(1)   0    0
        X(4) Y(4)          X(4) Y(4) X(2) Y(2)   0    0
        X(5) Y(5)          X(5) Y(5) X(3) Y(3) X(1) Y(1)
        X(6) Y(6)          ...  ...  ...  ...  ...  ... ]
        X(7) Y(7)
        X(8) Y(8)
        ...  ...  ]

    This node provides similar functionality as the ``TimeFramesNode``, only
    that it performs a time embedding into the past rather than into the future.

    See ``TimeDelaySlidingWindowNode`` for a sliding window delay node for
    application in a non-batch manner.

    Original code contributed by Sebastian Hoefer.
    Dec 31, 2010
    """

    def __init__(self, time_frames, gap=1, input_dim=None, dtype=None):
        """
        Input arguments:
        time_frames -- Number of delayed copies
        gap -- Time delay between the copies
        """
        super(TimeDelayNode, self).__init__(time_frames, gap,
                                            input_dim, dtype)

    def _execute(self, x):
        gap = self.gap
        rows = x.shape[0]
        cols = self.output_dim
        n = self.input_dim

        y = numx.zeros((rows, cols), dtype=self.dtype)

        for frame in range(self.time_frames):
            y[gap * frame:, frame * n:(frame + 1) * n] = x[:rows - gap * frame, :]

        return y

    def pseudo_inverse(self, y):
        raise NotImplementedError


class TimeDelaySlidingWindowNode(TimeDelayNode):
    """
    ``TimeDelaySlidingWindowNode`` is an alternative to ``TimeDelayNode``
    which should be used for online learning/execution. Whereas the
    ``TimeDelayNode`` works in a batch manner, for online application
    a sliding window is necessary which yields only one row per call.

    Applied to the same data the collection of all returned rows of the
    ``TimeDelaySlidingWindowNode`` is equivalent to the result of the
    ``TimeDelayNode``.

    Original code contributed by Sebastian Hoefer.
    Dec 31, 2010
    """

    def __init__(self, time_frames, gap=1, input_dim=None, dtype=None):
        """
        Input arguments:
        time_frames -- Number of delayed copies
        gap -- Time delay between the copies
        """

        self.time_frames = time_frames
        self.gap = gap
        super(TimeDelaySlidingWindowNode, self).__init__(time_frames, gap,
                                                         input_dim, dtype)
        self.sliding_wnd = None
        self.cur_idx = 0
        self.slide = False

    def _init_sliding_window(self):
        rows = self.gap + 1
        cols = self.input_dim * self.time_frames
        self.sliding_wnd = numx.zeros((rows, cols), dtype=self.dtype)

    def _execute(self, x):
        assert x.shape[0] == 1

        if self.sliding_wnd is None:
            self._init_sliding_window()

        gap = self.gap
        rows = self.sliding_wnd.shape[0]
        cols = self.output_dim
        n = self.input_dim

        new_row = numx.zeros(cols, dtype=self.dtype)
        new_row[:n] = x

        # Slide
        if self.slide:
            self.sliding_wnd[:-1, :] = self.sliding_wnd[1:, :]

        # Delay
        if self.cur_idx - gap >= 0:
            new_row[n:] = self.sliding_wnd[self.cur_idx - gap, :-n]

        # Add new row to matrix
        self.sliding_wnd[self.cur_idx, :] = new_row

        if self.cur_idx < rows - 1:
            self.cur_idx = self.cur_idx + 1
        else:
            self.slide = True

        return new_row[numx.newaxis, :]


class EtaComputerNode(Node):
    """Compute the eta values of the normalized training data.

    The delta value of a signal is a measure of its temporal
    variation, and is defined as the mean of the derivative squared,
    i.e. ``delta(x) = mean(dx/dt(t)^2)``.  ``delta(x)`` is zero if
    ``x`` is a constant signal, and increases if the temporal variation
    of the signal is bigger.

    The eta value is a more intuitive measure of temporal variation,
    defined as::
    
       eta(x) = T/(2*pi) * sqrt(delta(x))

    If ``x`` is a signal of length ``T`` which consists of a sine function
    that accomplishes exactly ``N`` oscillations, then ``eta(x)=N``.

    ``EtaComputerNode`` normalizes the training data to have unit
    variance, such that it is possible to compare the temporal
    variation of two signals independently from their scaling.

    Reference: Wiskott, L. and Sejnowski, T.J. (2002).
    Slow Feature Analysis: Unsupervised Learning of Invariances,
    Neural Computation, 14(4):715-770.

    Important: if a data chunk is tlen data points long, this node is
    going to consider only the first tlen-1 points together with their
    derivatives. This means in particular that the variance of the
    signal is not computed on all data points. This behavior is
    compatible with that of ``SFANode``.

    This is an analysis node, i.e. the data is analyzed during training
    and the results are stored internally.  Use the method
    ``get_eta`` to access them.
    """

    def __init__(self, input_dim=None, dtype=None):
        super(EtaComputerNode, self).__init__(input_dim, None, dtype)
        self._initialized = 0

    def _set_input_dim(self, n):
        self._input_dim = n
        self.output_dim = n

    def _init_internals(self):
        input_dim = self.input_dim
        self._mean = numx.zeros((input_dim,), dtype='d')
        self._var = numx.zeros((input_dim,), dtype='d')
        self._tlen = 0
        self._diff2 = numx.zeros((input_dim,), dtype='d')
        self._initialized = 1

    def _train(self, data):
        # here SignalNode.train makes an automatic refcast
        if not self._initialized:
            self._init_internals()

        rdata = data[:-1]
        self._mean += rdata.sum(axis=0)
        self._var += (rdata * rdata).sum(axis=0)
        self._tlen += rdata.shape[0]
        td_data = utils.timediff(data)
        self._diff2 += (td_data * td_data).sum(axis=0)

    def _stop_training(self):
        var_tlen = self._tlen - 1
        # unbiased
        var = old_div((self._var - self._mean * self._mean / self._tlen), var_tlen)

        # biased
        # var = (self._var - self._mean*self._mean/self._tlen)/self._tlen

        # old formula: wrong! is neither biased nor unbiased
        # var = (self._var/var_tlen) - (self._mean/self._tlen)**2

        self._var = var
        delta = old_div((old_div(self._diff2, self._tlen)), var)
        self._delta = delta
        self._eta = old_div(numx.sqrt(delta), (2 * numx.pi))

    def get_eta(self, t=1):
        """Return the eta values of the data received during the training
        phase. If the training phase has not been completed yet, call
        stop_training.

        :Arguments:
           t
             Sampling frequency in Hz.

             The original definition in (Wiskott and Sejnowski, 2002)
             is obtained for ``t=self._tlen``, while for ``t=1`` (default),
             this corresponds to the beta-value defined in
             (Berkes and Wiskott, 2005).
        """
        self._if_training_stop_training()
        return self._refcast(self._eta * t)


class NoiseNode(PreserveDimNode):
    """Inject multiplicative or additive noise into the input data.

    Original code contributed by Mathias Franzius.
    """

    def __init__(self, noise_func=mdp.numx_rand.normal, noise_args=(0, 1),
                 noise_type='additive',
                 input_dim=None, output_dim=None, dtype=None):
        """
        Add noise to input signals.

        :Arguments:
          noise_func
            A function that generates noise. It must
            take a ``size`` keyword argument and return
            a random array of that size. Default is normal noise.

          noise_args
            Tuple of additional arguments passed to `noise_func`.
            Default is (0,1) for (mean, standard deviation)
            of the normal distribution.

          noise_type
            Either ``'additive'`` or ``'multiplicative'``.

            'additive'
               returns ``x + noise``.
            'multiplicative'
               returns ``x * (1 + noise)``

            Default is ``'additive'``.
        """
        super(NoiseNode, self).__init__(input_dim=input_dim,
                                        output_dim=output_dim,
                                        dtype=dtype)
        self.noise_func = noise_func
        self.noise_args = noise_args
        valid_noise_types = ['additive', 'multiplicative']
        if noise_type not in valid_noise_types:
            err_str = '%s is not a valid noise type' % str(noise_type)
            raise NodeException(err_str)
        else:
            self.noise_type = noise_type

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        return (mdp.utils.get_dtypes('Float') +
                mdp.utils.get_dtypes('AllInteger'))

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    def _execute(self, x):
        noise_mat = self._refcast(self.noise_func(*self.noise_args,
                                                  **{'size': x.shape}))
        if self.noise_type == 'additive':
            return x + noise_mat
        elif self.noise_type == 'multiplicative':
            return x * (1. + noise_mat)

    def save(self, filename, protocol=-1):
        """Save a pickled serialization of the node to 'filename'.
        If 'filename' is None, return a string.

        Note: the pickled Node is not guaranteed to be upward or
        backward compatible."""
        if filename is None:
            # cPickle seems to create an error, probably due to the
            # self.noise_func attribute.
            return real_pickle.dumps(self, protocol)
        else:
            # if protocol != 0 open the file in binary mode
            mode = 'w' if protocol == 0 else 'wb'
            with open(filename, mode) as flh:
                real_pickle.dump(self, flh, protocol)


class NormalNoiseNode(PreserveDimNode):
    """Special version of ``NoiseNode`` for Gaussian additive noise.

    Unlike ``NoiseNode`` it does not store a noise function reference but simply
    uses ``numx_rand.normal``.
    """

    def __init__(self, noise_args=(0, 1),
                 input_dim=None, output_dim=None, dtype=None):
        """Set the noise parameters.

        noise_args -- Tuple of (mean, standard deviation) for the normal
            distribution, default is (0,1).
        """
        super(NormalNoiseNode, self).__init__(input_dim=input_dim,
                                              output_dim=output_dim,
                                              dtype=dtype)
        self.noise_args = noise_args

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    def _execute(self, x):
        noise = self._refcast(mdp.numx_rand.normal(size=x.shape) *
                              self.noise_args[1]
                              + self.noise_args[0])
        return x + noise


class CutoffNode(PreserveDimNode):
    """Node to cut off values at specified bounds.

    Works similar to ``numpy.clip``, but also works when only a lower or upper
    bound is specified.
    """

    def __init__(self, lower_bound=None, upper_bound=None,
                 input_dim=None, output_dim=None, dtype=None):
        """Initialize node.

        :Parameters:
          lower_bound
            Data values below this are cut to the ``lower_bound`` value.
            If ``lower_bound`` is ``None`` no cutoff is performed.
          upper_bound
            Works like ``lower_bound``.
        """
        super(CutoffNode, self).__init__(input_dim=input_dim,
                                         output_dim=output_dim,
                                         dtype=dtype)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    def _get_supported_dtypes(self):
        return (mdp.utils.get_dtypes('Float') +
                mdp.utils.get_dtypes('AllInteger'))

    def _execute(self, x):
        """Return the clipped data."""
        # n.clip() does not work, since it does not accept None for one bound
        if self.lower_bound is not None:
            x = numx.where(x >= self.lower_bound, x, self.lower_bound)
        if self.upper_bound is not None:
            x = numx.where(x <= self.upper_bound, x, self.upper_bound)
        return x


class HistogramNode(PreserveDimNode):
    """Node which stores a history of the data during its training phase.

    The data history is stored in ``self.data_hist`` and can also be deleted to
    free memory. Alternatively it can be automatically pickled to disk.

    Note that data is only stored during training.
    """

    def __init__(self, hist_fraction=1.0, hist_filename=None,
                 input_dim=None, output_dim=None, dtype=None):
        """Initialize the node.

        hist_fraction -- Defines the fraction of the data that is stored
            randomly.
        hist_filename -- Filename for the file to which the data history will
            be pickled after training. The data is pickled when stop_training
            is called and data_hist is then cleared (to free memory).
            If filename is None (default value) then data_hist is not cleared
            and can be directly used after training.
        """
        super(HistogramNode, self).__init__(input_dim=input_dim,
                                            output_dim=output_dim,
                                            dtype=dtype)
        self._hist_filename = hist_filename
        self.hist_fraction = hist_fraction
        self.data_hist = None  # stores the data history

    def _get_supported_dtypes(self):
        return (mdp.utils.get_dtypes('AllFloat') +
                mdp.utils.get_dtypes('AllInteger') +
                mdp.utils.get_dtypes('Character'))

    def _train(self, x):
        """Store the history data."""
        if self.hist_fraction < 1.0:
            x = x[numx.random.random(len(x)) < self.hist_fraction]
        if self.data_hist is not None:
            self.data_hist = numx.concatenate([self.data_hist, x])
        else:
            self.data_hist = x

    def _stop_training(self):
        """Pickle the histogram data to file and clear it if required."""
        super(HistogramNode, self)._stop_training()
        if self._hist_filename:
            pickle_file = open(self._hist_filename, "wb")
            try:
                pickle.dump(self.data_hist, pickle_file, protocol=-1)
            finally:
                pickle_file.close()
            self.data_hist = None


class AdaptiveCutoffNode(HistogramNode):
    """Node which uses the data history during training to learn cutoff values.

    As opposed to the simple ``CutoffNode``, a different cutoff value is learned
    for each data coordinate. For example if an upper cutoff fraction of
    0.05 is specified, then the upper cutoff bound is set so that the upper
    5% of the training data would have been clipped (in each dimension).
    The cutoff bounds are then applied during execution.
    This node also works as a ``HistogramNode``, so the histogram data is stored.

    When ``stop_training`` is called the cutoff values for each coordinate are
    calculated based on the collected histogram data.
    """

    def __init__(self, lower_cutoff_fraction=None, upper_cutoff_fraction=None,
                 hist_fraction=1.0, hist_filename=None,
                 input_dim=None, output_dim=None, dtype=None):
        """Initialize the node.

        :Parameters:
          lower_cutoff_fraction
            Fraction of data that will be cut off after
            the training phase (assuming the data distribution does not
            change). If set to ``None`` (default value) no cutoff is performed.
          upper_cutoff_fraction
            Works like `lower_cutoff_fraction`.
          hist_fraction
            Defines the fraction of the data that is stored for the
            histogram.
          hist_filename
            Filename for the file to which the data history will be
            pickled after training. The data is pickled when
            `stop_training` is called and ``data_hist`` is then
            cleared (to free memory).  If filename is ``None``
            (default value) then ``data_hist`` is not cleared and can
            be directly used after training.
        """
        super(AdaptiveCutoffNode, self).__init__(hist_fraction=hist_fraction,
                                                 hist_filename=hist_filename,
                                                 input_dim=input_dim,
                                                 output_dim=output_dim,
                                                 dtype=dtype)
        self.lower_cutoff_fraction = lower_cutoff_fraction
        self.upper_cutoff_fraction = upper_cutoff_fraction
        self.lower_bounds = None
        self.upper_bounds = None

    def _get_supported_dtypes(self):
        return (mdp.utils.get_dtypes('Float') +
                mdp.utils.get_dtypes('AllInteger'))

    def _stop_training(self):
        """Calculate the cutoff bounds based on collected histogram data."""
        if self.lower_cutoff_fraction or self.upper_cutoff_fraction:
            sorted_data = self.data_hist.copy()
            sorted_data.sort(axis=0)
            if self.lower_cutoff_fraction:
                index = self.lower_cutoff_fraction * len(sorted_data)
                self.lower_bounds = sorted_data[int(index)]
            if self.upper_cutoff_fraction:
                index = (len(sorted_data) -
                         self.upper_cutoff_fraction * len(sorted_data))
                self.upper_bounds = sorted_data[int(index)]
        super(AdaptiveCutoffNode, self)._stop_training()

    def _execute(self, x):
        """Return the clipped data."""
        if self.lower_bounds is not None:
            x = numx.where(x >= self.lower_bounds, x, self.lower_bounds)
        if self.upper_bounds is not None:
            x = numx.where(x <= self.upper_bounds, x, self.upper_bounds)
        return x


class NumxBufferNode(mdp.PreserveDimOnlineNode):
    """
    NumxBufferNode is an onlinenode that stores input data into a finite sized
    numpy/scipy array buffer.

    The train function updates the buffer with the input, whereas the execute
    function simply returns a copy of the buffer.

    This node also supports integer input data types.
    """

    def __init__(self, buffer_size, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """
        buffer_size - maximum buffer size (maximum number of samples the buffer can store)
        """
        super(NumxBufferNode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self._buffer_size = buffer_size
        self._buffer = None

    def _check_params(self, x):
        if self._buffer is None:
            self._buffer = mdp.numx.zeros((self._buffer_size, self.input_dim), dtype=self.dtype)

    def _get_supported_dtypes(self):
        return mdp.utils.get_dtypes('Float') + mdp.utils.get_dtypes('AllInteger')

    @staticmethod
    def is_invertible():
        return False

    def _train(self, x):
        if x.shape[0] > self._buffer_size:
            self._buffer = x[-self._buffer_size:].copy()
        else:
            self._buffer = mdp.numx.roll(self._buffer, -x.shape[0], axis=0)
            self._buffer[-x.shape[0]:] = x.copy()

    def _execute(self, x):
        return self._buffer.copy()


class TransformerNode(mdp.Node):
    """
    TransformerNode applies a sequence of transformations to the input.
    This node can be used to transform data processed between the nodes in a Flow.
    It is especially useful as a pre-processing node for image data before passing
    it to a training node such as an Hierarchical SFANode.

    The transformations to be applied are specified via the 'transform_seq' variable,
    which needs to be a list of "transformation function names".
    The transformations are applied on the input data from left-to-right of
    the sequence. The 'transform_seq' can be set either by passing it as an argument
    while creating the node or later via the property 'transform_seq'.

    The list of available transformation function names can be accessed by calling
    'get_transform_fn_names' method.

    To add other transformation methods, subclasses can overwrite the method
    '_get_transform_fns' that returns a dict mapping labels to the transformation methods.

    A few of the transformations might need additional arguments. Set 'transform_seq_args' variable equal
    to the list of argument-tuples in the same order as the 'transform_seq'. Use None whenever an
    intermediate transformation function requires no arguments. In any case, the 'transform_seq_args'
    setter raises an exception if the required number of arguments are not passed.

    An example implementation of the node:

    node = TransformerNode(input_shape=(640, 480, 3), transform_seq=['set_shape', 'gray', 'img_255_1', 'center'])

    Let the input to the node be a block of 10 flattened RGB images (shape - (10, 640*480*3)).
    The input is first reshaped to the given inp_shape since no argument has been passed (shape - (10, 640, 480, 3)).
    It is then converted to grayscale (shape - (10, 640, 480)).
    The output is then rescaled from [0, 255] to [0, 1], and then the data is centered to have zero mean.
    The final output returned is a flattened 2d array (shape - (10, 640*480*3)).

    The node also includes sklearn's pre-processing functions if sklearn package is available.

    """

    def __init__(self, input_shape, transform_seq=None, transform_seq_args=None, input_dim=None, dtype=None):
        """
        input_shape - The shape of each input sample. E.g. for a video of ten 640 x 480 images, the input_shape
                        is (640, 480). The input data is always first transformed to this shape.
        transform_seq - A list of strings, where each string represents a label for the transformation.
                        Supported transformation:
                            'transpose' - transposes axis dimensions > 0
                            'center' - centers the data (removes mean over axis=0)
                            'resize' - resize data dims (for image data). Requires python-opencv or pil packages.
                            'img_255_1' - scales uint img data to float values between [0, 1]
                            'gray' - converts to grayscale images.
                            'to_2d' - converts data to 2D
                            'to_dtype' - converts data to the desired data type
                            'set_shape' - reshapes the data to the given shape. If shape argument is not provided,
                            then it uses a stored shape buffer, which is either the input_shape or the shape
                            lost if 'to_2d' is executed.

        trasnform_seq_args - A list of required arguments tuples for each transformation, ordered according to
                            the transform_seq.
        """
        super(TransformerNode, self).__init__(input_dim=input_dim, output_dim=None, dtype=dtype)

        self.input_shape = input_shape
        self._input_dim = mdp.numx.product(self.input_shape)
        self._transform_seq = None
        self._transform_seq_args = None

        self.transform_seq = transform_seq
        self.transform_seq_args = transform_seq_args

        self._shape_buffer = input_shape

        # infer output_dim (easiest way)
        dummy_x = mdp.numx_rand.randn(1, mdp.numx.product(self.input_shape))
        self._output_dim = self._execute(dummy_x).shape[1]

    # properties

    @property
    def transform_seq(self):
        """Returns the list of transformation function names"""
        return self._transform_seq

    @transform_seq.setter
    def transform_seq(self, seq):
        """Sets the list of transformation function names"""
        if seq is None:
            return

        if self._transform_seq is not None:
            raise mdp.NodeException("'transform_seq' is already set to %s, "
                                    "given %s." % (str(self._transform_seq), str(seq)))

        # check transform sequence
        _transform_fns = self._get_transform_fns()
        for elem in seq:
            if elem not in _transform_fns.keys():
                mdp.NodeException("Unrecognized transform fn. Supported ones: %s" % str(_transform_fns.keys()))
        self._transform_seq = seq

    @property
    def transform_seq_args(self):
        """Returns the list of arguments for the transformation functions"""
        return self._transform_seq_args

    @transform_seq_args.setter
    def transform_seq_args(self, seq_args):
        """Sets the list of arguments for the transformation functions.
        Also checks if the required arguments are provided."""
        if seq_args is None:
            return

        if self._transform_seq_args is not None:
            raise mdp.NodeException("'transform_seq_args' is already set to "
                                    "%s, given %s." % (str(self._transform_seq_args), str(seq_args)))

        # check transform sequence args
        if (not isinstance(seq_args, list)) and (not isinstance(seq_args, tuple)):
            raise mdp.NodeException("'transform_seq_args' must be a list and not %s." % str(type(seq_args)))
        if len(self.transform_seq) != len(seq_args):
            raise mdp.NodeException("Wrong set of 'transform_seq_args', required "
                                    "%d given %d." % (len(self.transform_seq), len(seq_args)))
        for i, arg in enumerate(seq_args):
            key = self.transform_seq[i]
            fn_arg_keys = self._get_required_fn_args(key)
            fn_args_needed = bool(len(fn_arg_keys))
            if fn_args_needed:
                if len(arg) != len(fn_arg_keys):
                    err = ("Wrong number of arguments provided for the %s function " % str(key) +
                           "(%d needed, %d given).\n" % (len(fn_arg_keys), len(arg)) +
                           "List of required argument keys: " + str(fn_arg_keys))
                    raise mdp.NodeException(err)
            else:
                seq_args[i] = ()
        self._transform_seq_args = seq_args

    def _get_transform_fns(self):
        """Returns the transformation functions"""
        d = {'transpose': self._transpose, 'center': self._center, 'resize': self._resize,
             'img_255_1': self._img_255_1,
             'gray': self._gray, 'to_2d': self._to_2d, 'set_shape': self._set_shape, 'to_dtype': self._to_dtype}

        if config.has_sklearn:
            from sklearn import preprocessing
            _skd = {_key: getattr(preprocessing, _key) for _key in
                    filter(lambda a: a[0].islower(), preprocessing.__all__)}
            d.update(_skd)
        return d

    def _get_required_fn_args(self, fn_name):
        """Return arguments for transform function 'fn_name'
        Argumentes that have a default value are ignored.
        """
        _transform_fns = self._get_transform_fns()
        train_arg_spec = _inspect.getargspec(_transform_fns[fn_name])
        if train_arg_spec[0][0] == 'self':
            train_arg_keys = train_arg_spec[0][2:]  # ignore self, x
        else:
            # staticmethod
            train_arg_keys = train_arg_spec[0][1:]  # ignore just x
        if train_arg_spec[3]:
            # subtract arguments with a default value
            train_arg_keys = train_arg_keys[:-len(train_arg_spec[3])]
        return train_arg_keys

    # transformation methods

    @staticmethod
    def _transpose(x):
        """Transposes axis dimensions > 0

        E.g.: If input x has the shape = (10, 4, 5, 6),
        then the output will have the shape = (10, 6, 5, 4).

        This is useful for a fast transpose of multiple images or a video.

        """
        return mdp.numx.rollaxis(x.T, -1)

    @staticmethod
    def _center(x):
        """Centers the data (removes mean over axis 0)"""
        return x - x.mean(axis=0)

    @staticmethod
    def _resize(x, size_xy):
        """Resize data dims (for image data). Requires python-opencv or pil packages.

        E.g.: If the input x has the shape = (10, 6, 6) and the size_xy = (3, 3),
        then the output will have the shape = (10, 3, 3).
        """

        if config.has_opencv:
            from cv2 import resize
            rimg = [(resize(_x, size_xy[::-1])) for _x in x]
        elif config.has_pil:
            from PIL.Image import fromarray
            rimg = []
            for _x in x:
                if _x.dtype in ['uint64', 'int64', 'float64', 'complex64', 'complex128']:
                    raise mdp.NodeException("Unsupported dtype for resize. Use 'to_dtype' transformation method"
                                            "to convert to a dtype <=32 bit before 'resize'.")
                if len(_x.shape) == 3:
                    if _x.dtype in ['uint8', 'int8']:
                        imx = fromarray(_x, 'RGB')
                    else:
                        raise mdp.NodeException("Unsupported dtype for resize. Use 'to_dtype' transformation method"
                                                "to convert to a dtype ('uint8' or 'int8') before 'resize'.")
                elif _x.dtype in ['uint8', 'int8']:
                    imx = fromarray(_x, 'L')
                elif _x.dtype in mdp.utils.get_dtypes('UnsignedInteger') + mdp.utils.get_dtypes('Integer'):
                    imx = fromarray(_x, 'I')
                else:
                    imx = fromarray(_x, 'F')
                rimg.append(mdp.numx.array(imx.resize(size_xy[::-1])))
        else:
            raise mdp.NodeException("OpenCV python bindings or PIL Image package is required to resize images.")

        return mdp.numx.asarray(rimg)

    @staticmethod
    def _img_255_1(x):
        """Scales uint img data to float values between [0, 1]"""
        x = old_div(x, 255.)
        return x

    @staticmethod
    def _gray(x):
        """Converts the RGB image to grayscale.
        The last dimension is expected to represent the color channel
        """
        return x.mean(axis=-1)

    def _to_2d(self, x):
        """Reshapes data to 2D.
        Also, stores the input shape information in 'self._shape_buffer'.
        This can used to later restore the data's shape.
        """
        if x.ndim == 2:
            return x
        self._shape_buffer = x.shape[1:]
        return x.reshape(x.shape[0], mdp.numx.product(x.shape[1:]))

    @staticmethod
    def _to_dtype(x, dtype):
        """Changes the data type of the input.
        Useful when dealing with Image data.
        """
        return x.astype(dtype)

    def _set_shape(self, x, shape=None):
        """Reshapes the input data to the given shape.
        Useful for processing images. If given shape is None,
        the stored self._shape_buffer is used instead.
        """
        print self._shape_buffer
        if shape is None:
            # check if it can be reshapes with the stored shape buffer
            if mdp.numx.product(self._shape_buffer) == mdp.numx.product(x.shape[1:]):
                return x.reshape(x.shape[0], *self._shape_buffer)
            else:
                raise mdp.NodeException("Cannot reshape to the stored shape buffer. Provide the desired shape.")
        else:
            return x.reshape(x.shape[0], *shape)

    # Node methods

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    def _execute(self, x):
        if self.transform_seq is None:
            # no transformations
            return x
        elif self.transform_seq_args is None:
            # checks if no args needed
            self.transform_seq_args = [None] * len(self.transform_seq)

        _transform_fns = self._get_transform_fns()
        for i, fn_name in enumerate(self.transform_seq):
            x = _transform_fns[fn_name](x, *self.transform_seq_args[i])

        # reset shape buffer
        self._shape_buffer = self.input_shape

        # output to 2d
        x = x.reshape(x.shape[0], mdp.numx.product(x.shape[1:]))

        return self._refcast(x)

    # utility methods

    def get_transform_fn_names(self):
        """Returns the available transformation function names"""
        fns = self._get_transform_fns()
        return fns.keys()


# TODO: Could use or add to the graph package of MDP?
class GridProcessingNode(mdp.Node):
    """
    GridProcessingNode is a non-trainable node that
    translates an n-dimensional cartesian grid to a topological map
    of graph nodes labeled with n-dimensional coordinates and
    a 1D index. Therefore, the node provides conversions
    from grid coordinates to graph coordinates or graph indices.

    The type of conversion can be selected by setting the 'output_type'
    attribute. Accepted values:
    'gridx' - default (just returns the input)
    'graphx' - returns graph coordinates
    'graphindx' - returns graph indices of the input.

    The node is also invertible, enabling conversions
    back from 'graphx' and 'graphindx' to 'gridx'.

    Eg.

    3x3 cartesian grid points to  'graphx' output_type

    (0,0)   (2,0)   (4,0)       (0,0)   (1,0)   (2,0)

    (0,2)   (2,2)   (4,2)  <->  (0,1)   (1,1)   (2,1)

    (0,4)   (2,4)   (4,4)       (0,2)   (1,2)   (2,2)

    3x3 cartesian grid points to 'graphindx' output_type

    (0,0)   (2,0)   (4,0)       0   1   2

    (0,2)   (2,2)   (4,2)  <->  3   4   5

    (0,4)   (2,4)   (4,4)       6   7   8


    The node also provides a utility method:
     'get_neighbors' - Returns the neighboring
     grid points based on a distance function applied to
     the graph coordinates. The distance function
     can be specified using the 'nbr_dist_fn' argument
     and the threshold can be specified using the
     'nbr_radius' argument.

     'get_adjacency_matrix' - Returns an adjacency matrix
     of the graph.

    """

    def __init__(self, grid_lims, n_grid_pts=None, output_type=None, nbr_dist_fn=1, nbr_radius=1,
                 include_central_elem=True, input_dim=None, dtype=None):
        """
        grid_lims - a tuple of lower and upper bounds for each dimension of the input grid.
                Eg., [(lower1, lower2, ...,), (upper1, upper2, ...,)]

        n_grid_pts - number of grid points within each dimension. This provides the spacing between
                the grid points. Default is set to the number of integers between the bounds.

        output_type - the type of conversion. Accepted values:
                    'gridx' - default (just returns the input)
                    'graphx' - returns graph coordinates
                    'graphindx' - returns graph indices of the input.

        nbr_dist_fn - distance function to define the closest neighbors (applied on the graph
                    coordinates). This is the same as
                    "order" argument of the numx_linalg function. (Doc added here for quick reference)
                     Accepted values:
                    {non-zero int, inf, -inf, 'fro', 'nuc'}, inf means numpy's `inf` object.

                    For values of ``nbr_dist_fn <= 0``, the result is, strictly speaking, not a
                    mathematical 'norm', but it may still be useful for various numerical
                    purposes.

                    The following norms can be calculated:

                ==========  ============================  ==========================
                nbr_dist_fn    norm for matrices             norm for vectors
                ==========  ============================  ==========================
                    None   Frobenius norm                2-norm
                    'fro'  Frobenius norm                --
                    'nuc'  nuclear norm                  --
                    inf    max(sum(abs(x), axis=1))      max(abs(x))
                    -inf   min(sum(abs(x), axis=1))      min(abs(x))
                    0      --                            sum(x != 0)
                    1      max(sum(abs(x), axis=0))      as below
                    -1     min(sum(abs(x), axis=0))      as below
                    2      2-norm (largest sing. value)  as below
                    -2     smallest singular value       as below
                    other  --                            sum(abs(x)**ord)**(1./ord)
                    =====  ============================  ==========================

                For further info, refer to Numpy or Scipy linalg.norm documentation.

        nbr_radius - The radial distance to determine neighbors.

        include_central_elem - Includes distance 0 also as a neighour (that is, the grid point itself)

        """
        super(GridProcessingNode, self).__init__(input_dim, output_dim=None, dtype=dtype)
        if len(grid_lims) != 2:
            raise mdp.NodeException("'grid_lims' has %d elements given, required 2 "
                                    "[(lower1, lower2, ...,), (upper1, upper2, ...,)]" % (len(grid_lims)))
        if mdp.numx.isscalar(grid_lims[0]):
            grid_lims = [tuple((lim,)) for lim in grid_lims]
        if len(grid_lims[0]) != len(grid_lims[1]):
            raise mdp.NodeException("Length of lower_bounds ('grid_lims[0]=%d') does not match the length "
                                    "of the upper_bounds ('grid_lims[1]=%d)." % (len(grid_lims[0]), len(grid_lims[1])))
        self.grid_lims = mdp.numx.asarray(grid_lims)

        if n_grid_pts is None:
            n_grid_pts = [int(self.grid_lims[1][i] - self.grid_lims[0][i] + 1) for i in xrange(len(self.grid_lims[0]))]
        elif mdp.numx.isscalar(n_grid_pts):
            n_grid_pts = [n_grid_pts] * len(self.grid_lims[0])
        if len(n_grid_pts) != len(self.grid_lims[0]):
            raise mdp.NodeException("Length of 'n_grid_pts' (given = %d) does not match with the "
                                    "number of grid dimensions (%d)." % (len(n_grid_pts), len(self.grid_lims[0])))
        self.n_grid_pts = mdp.numx.asarray(n_grid_pts, dtype='int')

        self.nbr_dist_fn = nbr_dist_fn
        self.nbr_radius = nbr_radius
        self.include_central_elem = include_central_elem

        self._output_type = None
        self.output_type = output_type

        self._grid = [mdp.numx.linspace(self.grid_lims[0, i], self.grid_lims[1, i], self.n_grid_pts[i], endpoint=True)
                      for i in xrange(self.n_grid_pts.shape[0])]

        self._graph_dim = self.n_grid_pts.shape[0]
        self._tot_graph_nodes = mdp.numx.product(self.n_grid_pts)
        self._graph_lims = mdp.numx.array([mdp.numx.zeros(self._graph_dim), self.n_grid_pts - 1])
        self._graph_elem = mdp.numx.asarray(self._get_graph_elem(self._graph_dim,
                                                                 self.nbr_radius,
                                                                 self.nbr_dist_fn, self.include_central_elem))

    @staticmethod
    def _get_graph_elem(graph_dim, radius, nbr_dist_fn, inclued_central_elem):
        _a = list(itertools.product(range(-radius, radius + 1), repeat=graph_dim))
        if inclued_central_elem:
            a = filter(lambda x: (mdp.numx_linalg.norm(x, nbr_dist_fn) <= radius), _a)
        else:
            a = filter(lambda x: (mdp.numx_linalg.norm(x, nbr_dist_fn) <= radius)
                                 and not (mdp.numx.array(x) == 0).all(), _a)
        return a

    def _get_graph_neighbors(self, graphx):
        nbrs = filter(lambda x: (self._graph_lims[0, :] <= x).all()
                                and (x <= self._graph_lims[1, :]).all(), graphx + self._graph_elem)
        return mdp.numx.atleast_2d(nbrs)

    # properties

    @property
    def output_type(self):
        """Returns the output type."""
        return self._output_type

    @output_type.setter
    def output_type(self, typ):
        """Sets the output type."""
        if typ is None:
            typ = 'gridx'
        elif typ not in ['gridx', 'graphx', 'graphindx']:
            raise mdp.NodeException("'output_type' must be either of 'gridx', 'graphx' or "
                                    "'graphindx', given %s." % str(typ))
        self._output_type = typ
        if self._output_type == 'gridx':
            self._output_dim = self._input_dim
        elif self._output_type == 'graphx':
            self._output_dim = self._input_dim
        else:
            self._output_dim = 1

    # conversion methods

    def _gridx_to_graphx(self, gridx):
        graphx = mdp.numx.zeros(gridx.shape)
        for i in xrange(self._graph_dim):
            graphx[:, i] = mdp.numx.argmin(mdp.numx.absolute(self._grid[i] - gridx[:, i:i + 1]), axis=1)
        return graphx

    def _graphx_to_gridx(self, graphx):
        gridx = mdp.numx.zeros(graphx.shape)
        for i in xrange(self._graph_dim):
            gridx[:, i] = self._grid[i][graphx[:, i].astype('int')]
        return gridx

    def _graphx_to_graphindx(self, graphx):
        graphindx = mdp.numx.zeros([graphx.shape[0], 1])
        for i in xrange(self._graph_dim):
            graphindx = graphindx * self.n_grid_pts[i] + graphx[:, i:i + 1]
        return graphindx

    def _graphindx_to_graphx(self, graphindx):
        _graphindx = graphindx.astype('int')
        graphx = mdp.numx.zeros((graphindx.shape[0], self._graph_dim))
        for i in xrange(self._graph_dim):
            _d = int(mdp.numx.product(self.n_grid_pts[i + 1:]))
            graphx[:, i:i + 1] = old_div(_graphindx, _d)
            _graphindx %= _d
        return graphx

    def _gridx_to_graphindx(self, gridx):
        graphindx = mdp.numx.zeros([gridx.shape[0], 1])
        for i in xrange(self._graph_dim):
            graphindx = graphindx * self.n_grid_pts[i] + \
                        mdp.numx.argmin(mdp.numx.absolute(self._grid[i] - gridx[:, i:i + 1]), axis=1)[:, None]
        return graphindx

    def _graphindx_to_gridx(self, graphindx):
        _graphindx = graphindx.astype('int')
        gridx = mdp.numx.zeros((graphindx.shape[0], self._graph_dim))
        for i in xrange(self._graph_dim):
            _d = int(mdp.numx.product(self.n_grid_pts[i + 1:]))
            graphx_i = old_div(_graphindx, _d)
            _graphindx %= _d
            gridx[:, i:i + 1] = self._grid[i][graphx_i]
        return gridx

    @staticmethod
    def is_trainable():
        return False

    def _train(self, x):
        pass

    def _execute(self, x):
        y = x
        if self.output_type == 'graphx':
            y = self._gridx_to_graphx(x)
        elif self.output_type == 'graphindx':
            y = self._gridx_to_graphindx(x)
        return self._refcast(y)

    def _inverse(self, y):
        x = y
        if self.output_type == 'graphx':
            x = self._graphx_to_gridx(y)
        elif self.output_type == 'graphindx':
            x = self._graphindx_to_gridx(y)
        return self._refcast(x)

    # utility methods
    def get_neighbors(self, gridx):
        """Returns the neighboring grid points based on a distance function applied to the graph coordinates."""
        return self._graphx_to_gridx(self._get_graph_neighbors(self._gridx_to_graphx(gridx)))

    def get_adjacency_matrix(self):
        """Returns an adjacency matrix of the graph."""
        adj = mdp.numx.zeros([self._tot_graph_nodes, self._tot_graph_nodes])
        for i in xrange(adj.shape[0]):
            i_arr = mdp.numx.array([[i]]).astype('float')
            for j in self._graphx_to_graphindx(self._get_graph_neighbors(self._graphindx_to_graphx(i_arr))):
                adj[i, int(j)] = 1
        return self._refcast(adj)
