from builtins import str
from builtins import range
__docformat__ = "restructuredtext en"

import mdp
from mdp import numx, Node, NodeException, TrainingException
from mdp.utils import (mult, pinv, CovarianceMatrix,
                       VartimeCovarianceMatrix, QuadraticForm,
                       symeig, SymeigException, symeig_semidefinite_reg,
                       symeig_semidefinite_pca, symeig_semidefinite_svd,
                       symeig_semidefinite_ldl)
import warnings

_INC_ARG_WARNING1 = ('As time_dependence is not specified, argument dt '
                     'should be of length x.shape[0]-1.')
warnings.filterwarnings('always', _INC_ARG_WARNING1, mdp.MDPWarning)

SINGULAR_VALUE_MSG = '''
This usually happens if there are redundancies in the (expanded) training data.
There are several ways to deal with this issue:

  - Use more data.

  - Use another solver for the generalized eigenvalue problem.
    The default solver requires the covariance matrix to be strictly positive
    definite. Construct your node with e.g. rank_deficit_method='auto' to use
    a more robust solver that allows positive semidefinite covariance matrix.
    Available values for rank_deficit_method: none, auto, pca, reg, svd, ldl
    See mdp.utils.symeig_semidefinite for details on the available methods.

  - Add noise to the data. This can be done by chaining an additional NoiseNode
    in front of a troublesome SFANode. Noise levels do not have to be high.
    Note:
    You will get a somewhat similar effect by rank_deficit_method='reg'.
    This will be more efficient in execution phase.

  - Run training data through PCA. This can be done by chaining an additional
    PCA node in front of the troublesome SFANode. Use the PCA node to discard
    dimensions within your data with lower variance.
    Note:
    You will get the same result by rank_deficit_method='pca'.
    This will be more efficient in execution phase.
'''


class SFANode(Node):
    """Extract the slowly varying components from the input data.

    .. attribute:: avg

        Mean of the input data (available after training)

    .. attribute:: sf

        Matrix of the SFA filters (available after training)

    .. attribute:: d

        Delta values corresponding to the SFA components (generalized
        eigenvalues). [See the docs of the ``get_eta_values`` method for
        more information]

    .. admonition:: Reference

        More information about Slow Feature Analysis can be found in
        Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
        Learning of Invariances, Neural Computation, 14(4):715-770 (2002).
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 include_last_sample=True, rank_deficit_method='none'):
        """Initialize an object of type 'SFANode'.

        :param input_dim: The input dimensionality.
        :type input_dim: int

        :param output_dim: The output dimensionality.
        :type output_dim: int

        :param dtype: The datatype.
        :type dtype: numpy.dtype or str

        :param include_last_sample: If ``False`` the `train` method discards
            the last sample in every chunk during training when calculating
            the covariance matrix.
            The last sample is in this case only used for calculating the
            covariance matrix of the derivatives. The switch should be set
            to ``False`` if you plan to train with several small chunks. For
            example we can split a sequence (index is time)::

                x_1 x_2 x_3 x_4

            in smaller parts like this::

                x_1 x_2
                x_2 x_3
                x_3 x_4

            The SFANode will see 3 derivatives for the temporal covariance
            matrix, and the first 3 points for the spatial covariance matrix.
            Of course you will need to use a generator that *connects* the
            small chunks (the last sample needs to be sent again in the next
            chunk). If ``include_last_sample`` was True, depending on the
            generator you use, you would either get::

                x_1 x_2
                x_2 x_3
                x_3 x_4

            in which case the last sample of every chunk would be used twice
            when calculating the covariance matrix, or::

                x_1 x_2
                x_3 x_4

            in which case you loose the derivative between ``x_3`` and ``x_2``.

            If you plan to train with a single big chunk leave
            ``include_last_sample`` to the default value, i.e. ``True``.

            You can even change this behaviour during training. Just set the
            corresponding switch in the `train` method.
        :type include_last_sample: bool

        :param rank_deficit_method: Possible values: 'none' (default), 'reg', 'pca', 'svd', 'auto'
            If not 'none', the ``stop_train`` method solves the SFA eigenvalue
            problem in a way that is robust against linear redundancies in
            the input data. This would otherwise lead to rank deficit in the
            covariance matrix, which usually yields a
            SymeigException ('Covariance matrices may be singular').
            There are several solving methods implemented:

            reg  - works by regularization
            pca  - works by PCA
            svd  - works by SVD
            ldl  - works by LDL decomposition (requires SciPy >= 1.0)

            auto - (Will be: selects the best-benchmarked method of the above)
                   Currently it simply selects pca.

            Note: If you already received an exception
            SymeigException ('Covariance matrices may be singular')
            you can manually set the solving method for an existing node::

               sfa.set_rank_deficit_method('pca')

            That means,::

               sfa = SFANode(rank_deficit='pca')

            is equivalent to::

               sfa = SFANode()
               sfa.set_rank_deficit_method('pca')

            After such an adjustment you can run ``stop_training()`` again,
            which would save a potentially time-consuming rerun of all
            ``train()`` calls.
        :type rank_deficit_method: str
        """
        super(SFANode, self).__init__(input_dim, output_dim, dtype)
        self._include_last_sample = include_last_sample

        # init two covariance matrices
        self._init_cov()

        # set routine for eigenproblem
        self.set_rank_deficit_method(rank_deficit_method)
        self.rank_threshold = 1e-12
        self.rank_deficit = 0

        # SFA eigenvalues and eigenvectors, will be set after training
        self.d = None
        self.sf = None  # second index for outputs
        self.avg = None
        self._bias = None  # avg multiplied with sf
        self.tlen = None

    def _init_cov(self):
        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(self.dtype)
        # one for the input data
        self._dcov_mtx = CovarianceMatrix(self.dtype)

    def set_rank_deficit_method(self, rank_deficit_method):
        if rank_deficit_method == 'pca':
            self._symeig = symeig_semidefinite_pca
        elif rank_deficit_method == 'reg':
            self._symeig = symeig_semidefinite_reg
        elif rank_deficit_method == 'svd':
            self._symeig = symeig_semidefinite_svd
        elif rank_deficit_method == 'ldl':
            try:
                from scipy.linalg.lapack import dsytrf
            except ImportError:
                err_msg = ("ldl method for solving SFA with rank deficit covariance "
                           "requires at least SciPy 1.0.")
                raise NodeException(err_msg)
            self._symeig = symeig_semidefinite_ldl
        elif rank_deficit_method == 'auto':
            self._symeig = symeig_semidefinite_pca
        elif rank_deficit_method == 'none':
            self._symeig = symeig
        else:
            raise ValueError("Invalid value for rank_deficit_method: %s"
                             % str(rank_deficit_method))

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative

        :param x: The time series data.
        :type x: numpy.ndarray

        :returns: Piecewise linear approximation of the time derivative.
        :rtype: numpy.ndarray
        """
        # this is faster than a linear_filter or a weave-inline solution
        return x[1:, :]-x[:-1, :]

    def _set_range(self):
        if self.output_dim is not None and self.output_dim <= self.input_dim:
            # (eigenvalues sorted in ascending order)
            rng = (1, self.output_dim)
        else:
            # otherwise, keep all output components
            rng = None
            self.output_dim = self.input_dim
        return rng

    def _check_train_args(self, x, *args, **kwargs):
        """Raises exception if time dimension does not have enough elements.

        :param x: The time series data.
        :type x: numpy.ndarray

        :param *args:
        :param **kwargs:
        """
        # check that we have at least 2 time samples to
        # compute the update for the derivative covariance matrix
        s = x.shape[0]
        if s < 2:
            raise TrainingException('Need at least 2 time samples to '
                                    'compute time derivative (%d given)' % s)

    def _train(self, x, include_last_sample=None):
        """Training method.

        :param x: The time series data.
        :type x: numpy.ndarray

        :param include_last_sample: For the ``include_last_sample`` switch have a
            look at the SFANode.__init__ docstring.
        :type include_last_sample: bool
        """
        if include_last_sample is None:
            include_last_sample = self._include_last_sample
        # works because x[:None] == x[:]
        last_sample_index = None if include_last_sample else -1

        # update the covariance matrices
        self._cov_mtx.update(x[:last_sample_index, :])
        self._dcov_mtx.update(self.time_derivative(x))

    def _stop_training(self, debug=False):
        # request the covariance matrices and clean up
        if hasattr(self, '_dcov_mtx'):
            self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
            del self._cov_mtx
        # do not center around the mean:
        # we want the second moment matrix (centered about 0) and
        # not the second central moment matrix (centered about the mean), i.e.
        # the covariance matrix
        if hasattr(self, '_dcov_mtx'):
            self.dcov_mtx, self.davg, self.dtlen = self._dcov_mtx.fix(
                    center=False)
            del self._dcov_mtx

        rng = self._set_range()

        # solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        try:
            try:
                # We first try to fulfill the extended signature described
                # in mdp.utils.symeig_semidefinite
                self.d, self.sf = self._symeig(
                        self.dcov_mtx, self.cov_mtx,
                        True, "on", rng,
                        overwrite=(not debug),
                        rank_threshold=self.rank_threshold,
                        dfc_out=self)
            except TypeError:
                self.d, self.sf = self._symeig(
                        self.dcov_mtx, self.cov_mtx, True,
                        "on", rng, overwrite=(not debug))
            d = self.d
            # check that we get only *positive* eigenvalues
            if d.min() < 0:
                err_msg = ("Got negative eigenvalues: %s.\n"
                           "You may either set output_dim to be smaller,\n"
                           "or prepend the SFANode with a PCANode(reduce=True)\n"
                           "or PCANode(svd=True)\n"
                           "or set a rank deficit method, e.g.\n"
                           "create the SFA node with rank_deficit_method='auto'\n"
                           "and try higher values for rank_threshold, e.g. try\n"
                           "your_node.rank_threshold = 1e-10, 1e-8, 1e-6, ..." % str(d))
                raise NodeException(err_msg)
        except SymeigException as exception:
            errstr = (str(exception)+"\n Covariance matrices may be singular.\n"
                      + SINGULAR_VALUE_MSG)
            raise NodeException(errstr)

        if not debug:
            # delete covariance matrix if no exception occurred
            del self.cov_mtx
            del self.dcov_mtx

        # store bias
        self._bias = mult(self.avg, self.sf)

    def _execute(self, x, n=None):
        """Compute the output of the slowest functions.

        :param x: The time series data.
        :type x: numpy.ndarray

        :param n: The number of slowest components. If 'n' is an integer,
            then use the first 'n' slowest components.
        :type n: int

        :returns: The output of the slowest functions.
        :rtype: numpy.ndarray
        """
        if n:
            sf = self.sf[:, :n]
            bias = self._bias[:n]
        else:
            sf = self.sf
            bias = self._bias
        return mult(x, sf) - bias

    def _inverse(self, y):
        return mult(y, pinv(self.sf)) + self.avg

    def get_eta_values(self, t=1):
        """Return the eta values of the slow components learned during
        the training phase. If the training phase has not been completed
        yet, call `stop_training`.

        The delta value of a signal is a measure of its temporal
        variation, and is defined as the mean of the derivative squared,
        i.e. delta(x) = mean(dx/dt(t)^2).  delta(x) is zero if
        x is a constant signal, and increases if the temporal variation
        of the signal is bigger.

        The eta value is a more intuitive measure of temporal variation,
        defined as
        eta(x) = t/(2*pi) * sqrt(delta(x))
        If x is a signal of length 't' which consists of a sine function
        that accomplishes exactly N oscillations, then eta(x)=N.

        :param t: Sampling frequency in Hz.

            The original definition in (Wiskott and Sejnowski, 2002)
            is obtained for t = number of training data points, while
            for t=1 (default), this corresponds to the beta-value defined
            in (Berkes and Wiskott, 2005).

        :returns: The eta values of the slow components learned during
            the training phase.
        """
        if self.is_training():
            self.stop_training()
        return self._refcast(t / (2 * numx.pi) * numx.sqrt(self.d))


class VartimeSFANode(SFANode):
    """Extract the slowly varying components from the input data.
    This node can be understood as a generalization of the *SFANode* that
    allows non-constant time increments between samples.

    In particular, this node numerically computes the integrals involved in
    the SFA problem formulation by applying the trapezoid rule.

    .. attribute:: avg

        Mean of the input data (available after training)

    .. attribute:: sf

        Matrix of the SFA filters (available after training)

    .. attribute:: d

        Delta values corresponding to the SFA components (generalized
        eigenvalues). [See the docs of the ``get_eta_values`` method for
        more information]

    .. admonition:: Reference

        More information about Slow Feature Analysis can be found in
        Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
        Learning of Invariances, Neural Computation, 14(4):715-770 (2002).
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 rank_deficit_method='none'):
        """Initialize an object of type 'VartimeSFANode'.

        :param input_dim: The input dimensionality.
        :type input_dim: int

        :param output_dim: The output dimensionality.
        :type output_dim: int

        :param dtype: The datatype.
        :type dtype: numpy.dtype or str

        :param rank_deficit_method: Possible values: 'none' (default), 'reg', 'pca', 'svd', 'auto'
            If not 'none', the ``stop_train`` method solves the SFA eigenvalue
            problem in a way that is robust against linear redundancies in
            the input data. This would otherwise lead to rank deficit in the
            covariance matrix, which usually yields a
            SymeigException ('Covariance matrices may be singular').
            There are several solving methods implemented:

            reg  - works by regularization
            pca  - works by PCA
            svd  - works by SVD
            ldl  - works by LDL decomposition (requires SciPy >= 1.0)

            auto - (Will be: selects the best-benchmarked method of the above)
                   Currently it simply selects pca.

            Note: If you already received an exception
            SymeigException ('Covariance matrices may be singular')
            you can manually set the solving method for an existing node::

               sfa.set_rank_deficit_method('pca')

            That means,::

               sfa = SFANode(rank_deficit='pca')

            is equivalent to::

               sfa = SFANode()
               sfa.set_rank_deficit_method('pca')

            After such an adjustment you can run ``stop_training()`` again,
            which would save a potentially time-consuming rerun of all
            ``train()`` calls.
        :type rank_deficit_method: str
        """
        super(VartimeSFANode, self).__init__(
                input_dim, output_dim, dtype, include_last_sample=True,
                rank_deficit_method=rank_deficit_method)

        self.tchunk = 0

    def _init_cov(self):
        # init two covariance matrices
        # one for the input data
        self._cov_mtx = VartimeCovarianceMatrix(self.dtype)
        # one for the derivatives
        self._dcov_mtx = VartimeCovarianceMatrix(self.dtype)

    def time_derivative(self, x, dt=None, time_dep=True):
        """Compute the linear approximation of the time derivative

        :param x: The time series data.
        :type x: numpy.ndarray

        :param dt: Sequence of time increments between vectors. 

            Usage with only single chunk of data:
                *dt* should be of length *x.shape[0]-1* or a constant. When
                constant, the time increments are assumed to be constant. If
                *dt* is not supplied, a constant time increment of one is
                assumed. If a time sequence of length *x.shape[0]* is supplied
                the first element is disregarded and a warning is presented.

            Usage with multiple chunks of data with intended time dependence:
                *dt* should have length *x.shape[0]-1* in the first call and
                be of length *x.shape[0]* starting with the second call.
                Starting with the second call, the first element in each chunk
                will be considered the time difference between the last element
                of *x* in the previous chunk and the first element of *x* of the
                current chunk.
                The *time_dep* argument being *True* indicates this mode.

            Usage with multiple chunks without time dependence:
                The moments are computed as a weighted average of the moments
                of separate chunks, if *dt* continues to have length
                *x.shape[0]-1* after the first call. Time dependence between
                chunks is thus omitted and all algorithmic components regard
                only the time structure within chunks.
                The *time_dep* argument being *False* indicates this mode.
                As in the single chunk case it is possible to set *dt* to be a
                constant or omit it completely for constant or unit time
                increments within chunks respectively.

        :type dt: numpy.ndarray or numeric

        :param time_dep: Indicates whether time dependence between chunks can
            be considered. The argument is only relevant if multiple chunks
            of data are used. Time dependence between chunks is disregarded
            when data collection has been done time-independently and thus no
            reasonable time increment between the end of a chunk and
            beginning of the next can be specified.
        
        :type time_dep: bool

        :returns: Piecewise linear approximation of the time derivative.
        :rtype: numpy.ndarray
        """
        if dt is not None and type(dt) == numx.ndarray:
            # check for inconsistent arguments
            if time_dep and self.tchunk >0 and x.shape[0]-1 == len(dt):
                raise Exception('As time_dependence is specified, and it is not the first'
                        '\ncall argument dt should be of length x.shape[0].')
            if (not time_dep or self.tchunk == 0) and x.shape[0] == len(dt):
                warnings.warn(_INC_ARG_WARNING1, mdp.MDPWarning)
            if len(dt) != x.shape[0] and len(dt) != x.shape[0]-1:
                raise Exception('Unexpected length of dt.')
        elif dt is not None and not dt > 0:
            raise Exception('Unexpected type or value of dt.')

        if dt is None:
            outlen = x.shape[0] if time_dep and self.tchunk > 0 else x.shape[0]-1
        elif type(dt) == numx.ndarray:
            outlen = x.shape[0]-1  if self.tchunk == 0 or dt.shape[0] != x.shape[0] else x.shape[0]
        elif dt > 0:
            outlen = x.shape[0]  if self.tchunk > 0  and time_dep else x.shape[0]-1
        out = numx.empty([outlen, x.shape[1]])

        if dt is not None and type(dt) == numx.ndarray:
            # Improvements can be made, by interpolating polynomials
            out[-x.shape[0]+1:, :] =\
                    (x[1:, :]-x[:-1, :]) / dt[-x.shape[0]+1:, None]
        elif dt is not None and dt > 0:
            out[-x.shape[0]+1:, :] = (x[1:, :]-x[:-1, :])/dt
        else:
            # trivial fallback
            out[-x.shape[0]+1:, :] = x[1:, :]-x[:-1, :]

        if self.tchunk > 0 and time_dep:
            if dt is None:
                out[0, :] = x[0, :]-self.xlast
            elif type(dt) == numx.ndarray and dt.shape[0] == x.shape[0]:
                out[0, :] = numx.divide(x[0, :]-self.xlast, dt[0])
            elif dt > 0:
                out[0, :] = numx.divide(x[0, :]-self.xlast, dt)

        self.xlast = x[-1, :].copy()
        self.tchunk += 1

        return out

    def _train(self, x, dt=None, time_dep=True):
        """Training method.

        :param x: The time series data.
        :type x: numpy.ndarray

        :param dt: Sequence of time increments between vectors. 

            Usage with only single chunk of data:
                *dt* should be of length *x.shape[0]-1* or a constant. When
                constant, the time increments are assumed to be constant. If
                *dt* is not supplied, a constant time increment of one is
                assumed. If a time sequence of length *x.shape[0]* is supplied
                the first element is disregarded and a warning is presented.

            Usage with multiple chunks of data with intended time dependence:
                *dt* should have length *x.shape[0]-1* in the first call and
                be of length *x.shape[0]* starting with the second call.
                Starting with the second call, the first element in each chunk
                will be considered the time difference between the last element
                of *x* in the previous chunk and the first element of *x* of the
                current chunk.
                The *time_dep* argument being *True* indicates this mode.

            Usage with multiple chunks without time dependence:
                The moments are computed as a weighted average of the moments
                of separate chunks, if *dt* continues to have length
                *x.shape[0]-1* after the first call. Time dependence between
                chunks is thus omitted and all algorithmic components regard
                only the time structure within chunks.
                The *time_dep* argument being *False* indicates this mode.
                As in the single chunk case it is possible to set *dt* to be a
                constant or omit it completely for constant or unit time
                increments within chunks respectively.

        :type dt: numpy.ndarray or numeric

        :param time_dep: Indicates whether time dependence between chunks can
            be considered. The argument is only relevant if multiple chunks
            of data are used. Time dependence between chunks is disregarded
            when data collection has been done time-independently and thus no
            reasonable time increment between the end of a chunk and
            beginning of the next can be specified.
        
        :type time_dep: bool
        """
        # update the covariance matrices
        self._cov_mtx.update(x, dt, time_dep)
        x_ = self.time_derivative(x, dt, time_dep)

        if (x_.shape[0] > x.shape[0]-1) and (dt is not None):
            dt_ = dt
        elif dt is not None and type(dt) == numx.ndarray:
            dt_ = dt[1:]
        elif dt is not None and dt > 0:
            dt_ = dt
        else:
            dt_ = None

        self._dcov_mtx.update(x_, dt_, time_dep)


class SFA2Node(SFANode):
    """Get an input signal, expand it in the space of
    inhomogeneous polynomials of degree 2 and extract its slowly varying
    components.

     The ``get_quadratic_form`` method returns the input-output
    function of one of the learned unit as a ``QuadraticForm`` object.
    See the documentation of ``mdp.utils.QuadraticForm`` for additional
    information.

    .. admonition:: Reference:

        More information about Slow Feature Analysis can be found in
        Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
        Learning of Invariances, Neural Computation, 14(4):715-770 (2002).
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 include_last_sample=True, rank_deficit_method='none'):
        """Initialize an object of type SFA2Node.

        :param input_dim: The input dimensionality.
        :type input_dim: int

        :param output_dim: The output dimensionality.
        :type output_dim: int

        :param dtype: The datatype.
        :type dtype: numpy.dtype or str

        :param include_last_sample: If ``False`` the `train` method discards the 
            last sample in every chunk during training when calculating 
            the covariance matrix.
            The last sample is in this case only used for calculating the
            covariance matrix of the derivatives. The switch should be set
            to ``False`` if you plan to train with several small chunks.
            For an example, see the SFANode.__init__ method's docstring.
        :type include_last_sample: bool

        :param rank_deficit_method: Possible values: 'none' (default), 'reg', 'pca', 'svd', 'auto'
            If not 'none', the ``stop_train`` method solves the SFA eigenvalue
            problem in a way that is robust against linear redundancies in
            the input data. This would otherwise lead to rank deficit in the
            covariance matrix, which usually yields a
            SymeigException ('Covariance matrices may be singular').
            For a more detailed description, have a look at the SFANode's constructor docstring.
        :type rank_deficit_method: str
        """
        self._expnode = mdp.nodes.QuadraticExpansionNode(input_dim=input_dim,
                                                         dtype=dtype)
        super(SFA2Node, self).__init__(input_dim, output_dim, dtype,
                                       include_last_sample, rank_deficit_method)

    @staticmethod
    def is_invertible():
        """Return True if the node can be inverted, False otherwise."""
        return False

    def _set_input_dim(self, n):
        self._expnode.input_dim = n
        self._input_dim = n

    def _train(self, x, include_last_sample=None):
        # expand in the space of polynomials of degree 2
        super(SFA2Node, self)._train(self._expnode(x), include_last_sample)

    def _set_range(self):
        if (self.output_dim is not None) and (
                self.output_dim <= self._expnode.output_dim):
            # (eigenvalues sorted in ascending order)
            rng = (1, self.output_dim)
        else:
            # otherwise, keep all output components
            rng = None
        return rng

    def _stop_training(self, debug=False):
        super(SFA2Node, self)._stop_training(debug)

        # set the output dimension if necessary
        if self.output_dim is None:
            self.output_dim = self._expnode.output_dim

    def _execute(self, x, n=None):
        """Compute the output of the slowest functions.

        :param x: The time series data.
        :type x: numpy.ndarray

        :param n: The number of slowest components. If 'n' is an integer,
            then use the first 'n' slowest components.
        :type n: int

        :returns: The output of the slowest functions.
        """
        return super(SFA2Node, self)._execute(self._expnode(x), n)

    def get_quadratic_form(self, nr):
        """Return the matrix H, the vector f and the constant c of the
        quadratic form 1/2 x'Hx + f'x + c that defines the output
        of the component 'nr' of the SFA node.

        :param nr: The component 'nr' of the SFA node.

        :returns: The matrix H, the vector f and the constant c of the
            quadratic form.
        :rtype: numpy.ndarray, numpy.ndarray, float
        """
        if self.sf is None:
            self._if_training_stop_training()

        sf = self.sf[:, nr]
        c = -mult(self.avg, sf)
        n = self.input_dim
        f = sf[:n]
        h = numx.zeros((n, n), dtype=self.dtype)
        k = n
        for i in range(n):
            for j in range(n):
                if j > i:
                    h[i, j] = sf[k]
                    k = k+1
                elif j == i:
                    h[i, j] = 2*sf[k]
                    k = k+1
                else:
                    h[i, j] = h[j, i]

        return QuadraticForm(h, f, c, dtype=self.dtype)


### old weave inline code to perform the time derivative

# weave C code executed in the function SfaNode.time_derivative
## _TDERIVATIVE_1ORDER_CCODE = """
##   for( int i=0; i<columns; i++ ) {
##     for( int j=0; j<rows-1; j++ ) {
##       deriv(j,i) = x(j+1,i)-x(j,i);
##     }
##   }
## """

# it was called like that:
## def time_derivative(self, x):
##     rows = x.shape[0]
##     columns = x.shape[1]
##     deriv = numx.zeros((rows-1, columns), dtype=self.dtype)

##     weave.inline(_TDERIVATIVE_1ORDER_CCODE,['rows','columns','deriv','x'],
##                  type_factories = weave.blitz_tools.blitz_type_factories,
##                  compiler='gcc',extra_compile_args=['-O3']);
##     return deriv
