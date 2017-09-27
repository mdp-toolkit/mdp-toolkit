from builtins import str
from builtins import range
__docformat__ = "restructuredtext en"

import mdp
from mdp import numx, Node, NodeException, TrainingException
from mdp.utils import (mult, svd, pinv, CovarianceMatrix, QuadraticForm,
                       symeig, SymeigException)

class SFANode(Node):
    """Extract the slowly varying components from the input data.
    More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
    Learning of Invariances, Neural Computation, 14(4):715-770 (2002).

    **Instance variables of interest**

      ``self.avg``
          Mean of the input data (available after training)

      ``self.sf``
          Matrix of the SFA filters (available after training)

      ``self.d``
          Delta values corresponding to the SFA components (generalized
          eigenvalues). [See the docs of the ``get_eta_values`` method for
          more information]

      ``self.rank_deficit``
          If an SFA solver detects rank deficit in the covariance matrix,
          it stores the count of zero eigenvalues as ``self.rank_deficit``.

    **Special arguments for constructor**

      ``include_last_sample``
          If ``False`` the `train` method discards the last sample in every
          chunk during training when calculating the covariance matrix.
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


      ``rank_deficit_method``
          Possible values: 'none' (default), 'reg', 'pca', 'svd', 'auto'
          If not ``none`` the `stop_train` method solves the SFA eigenvalue
          problem in a way that is robust against linear redundancies in
          the input data. This would otherwise lead to rank deficit in the
          covariance matrix, which usually yields a
          SymeigException ('Covariance matrices may be singular').
          There are several solving methods implemented:

          reg  - works by regularization
          pca  - works by PCA
          svd  - works by svd

          auto - selects the best-benchmarked method of the above

          Note: If you already received an exception
          SymeigException ('Covariance matrices may be singular')
          you can manually set the solving method for an existing node:

             sfa._sfa_solver = sfa._rank_deficit_solver_pca

          That means,

             sfa = SFANode(rank_deficit='pca')

          is equivalent to

             sfa = SFANode()
             sfa._sfa_solver = sfa._rank_deficit_solver_pca

          After such an adjustment you can run stop_training() again,
          which would save potentially time consuming rerun of all
          train() calls.
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 include_last_sample=True, rank_deficit_method='none'):
        """
        For the ``include_last_sample`` switch have a look at the
        SFANode class docstring.
        """
        super(SFANode, self).__init__(input_dim, output_dim, dtype)
        self._include_last_sample = include_last_sample

        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(dtype)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(dtype)

        # set routine for eigenproblem
        self._symeig = symeig
        if rank_deficit_method == 'pca':
            self._sfa_solver = self._rank_deficit_solver_pca
        elif rank_deficit_method == 'reg':
            self._sfa_solver = self._rank_deficit_solver_reg
        elif rank_deficit_method == 'svd':
            self._sfa_solver = self._rank_deficit_solver_svd
        elif rank_deficit_method == 'auto':
            self._sfa_solver = self._rank_deficit_solver_pca
        else:
            self._sfa_solver = None
        self.rank_threshold = 1e-12
        self.rank_deficit = 0

        # SFA eigenvalues and eigenvectors, will be set after training
        self.d = None
        self.sf = None  # second index for outputs
        self.avg = None
        self._bias = None  # avg multiplied with sf
        self.tlen = None

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative."""
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
        # check that we have at least 2 time samples to
        # compute the update for the derivative covariance matrix
        s = x.shape[0]
        if  s < 2:
            raise TrainingException('Need at least 2 time samples to '
                                    'compute time derivative (%d given)'%s)
        
    def _train(self, x, include_last_sample=None):
        """
        For the ``include_last_sample`` switch have a look at the
        SFANode class docstring.
        """
        if include_last_sample is None:
            include_last_sample = self._include_last_sample
        # works because x[:None] == x[:]
        last_sample_index = None if include_last_sample else -1

        # update the covariance matrices
        self._cov_mtx.update(x[:last_sample_index, :])
        self._dcov_mtx.update(self.time_derivative(x))

    def _stop_training(self, debug=False):
        ##### request the covariance matrices and clean up
        if hasattr(self, '_dcov_mtx'):
            self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
            del self._cov_mtx
        # do not center around the mean:
        # we want the second moment matrix (centered about 0) and
        # not the second central moment matrix (centered about the mean), i.e.
        # the covariance matrix
        if hasattr(self, '_dcov_mtx'):
            self.dcov_mtx, self.davg, self.dtlen = self._dcov_mtx.fix(center=False)
            del self._dcov_mtx

        rng = self._set_range()

        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        if self._sfa_solver is None:
            # We do not initialize this default method in the constructor to keep
            # code that inserts a custom _symeig (after the constructor) workable.
            self._sfa_solver = self._symeig
        try:
            self.d, self.sf = self._sfa_solver(
                    self.dcov_mtx, self.cov_mtx, True, "on", rng,
                    overwrite=(not debug))
            d = self.d
            # check that we get only *positive* eigenvalues
            if d.min() < 0:
                err_msg = ("Got negative eigenvalues: %s."
                           " You may either set output_dim to be smaller,"
                           " or prepend the SFANode with a PCANode(reduce=True)"
                           " or PCANode(svd=True)"% str(d))
                raise NodeException(err_msg)
        except SymeigException as exception:
            errstr = (str(exception)+"\n Covariance matrices may be singular."
                    +"\n Try to create SFA node with handle_rank_deficit=True.")
            raise NodeException(errstr)

        if not debug:
            # delete covariance matrix if no exception occurred
            del self.cov_mtx
            del self.dcov_mtx

        # store bias
        self._bias = mult(self.avg, self.sf)

    def _execute(self, x, n=None):
        """Compute the output of the slowest functions.
        If 'n' is an integer, then use the first 'n' slowest components."""
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

        :Parameters:
           t
             Sampling frequency in Hz.

             The original definition in (Wiskott and Sejnowski, 2002)
             is obtained for t = number of training data points, while
             for t=1 (default), this corresponds to the beta-value defined in
             (Berkes and Wiskott, 2005).
        """
        if self.is_training():
            self.stop_training()
        return self._refcast(t / (2 * numx.pi) * numx.sqrt(self.d))

    def _rank_deficit_solver_reg(
            self, A, B = None, eigenvectors=True, turbo="on", range=None,
            type=1, overwrite=False):
        """
        Alternative routine to solve the SFA eigenvalue issue. This can be used
        in case the normal symeig() call in _stop_training() throws the common
        SymeigException ('Covariance matrices may be singular').

        This solver applies a moderate regularization to the covariance matrix
        before applying symeig(). Afterwards it properly detects the rank
        deficit and filters out malformed features.
        This procedure is (approximately) as efficient as the ordinary SFA
        implementation based on plain symeig, because all additional steps are
        computationally cheap.

        Note:
        For efficiency reasons it actually modifies the covariance matrix
        (even if overwrite=False), but the changes are negligible.
        """
        if type != 1:
            raise ValueError('Only type=1 is supported.')

        # apply some regularization...
        # The following is equivalent to B += 1e-14*np.eye(B.shape[0]),
        # but works more in place, i.e. saves memory consumption of np.eye().
        Bflat = B.reshape(B.shape[0]*B.shape[1])
        idx = numx.arange(0, len(Bflat), B.shape[0]+1)
        diag_tmp = Bflat[idx]
        Bflat[idx] += self.rank_threshold

        eg, ev = self._symeig(A, B, True, turbo, None, type, overwrite)
        
        Bflat[idx] = diag_tmp
        m = numx.absolute(numx.sqrt(numx.sum(ev * mult(B, ev), 0))-1)
        off = 0
        # In theory all values in m should be close to one or close to zero.
        # So we use the mean of these values as threshold to distinguish cases:
        while m[off] > 0.5:
            off += 1
        m_off_sum = numx.sum(m[off:])
        if m_off_sum < 0.5:
            if off > 0:
                self.rank_deficit = off
                eg = eg[off:]
                ev = ev[:, off:]
        else:
            # Sometimes (unlikely though) the values in m are not sorted
            # In this case we search search all indices:
            count = int(round(m_off_sum))
            m_idx = numx.arange(len(m)-off-count)
            self.rank_deficit = off+count
            i = off
            while off < len(m_idx):
                if m[i] < 0.5:
                    m_idx[off] = i
                    off += 1
                i += 1
            eg = eg[m_idx]
            ev = ev[:, m_idx]
        if range is None:
            # self.d, self.sf
            return eg, ev
        else:
            # self.d, self.sf
            return eg[range[0]-1:range[1]], ev[:, range[0]-1:range[1]]

    def _rank_deficit_solver_pca(
            self, A, B = None, eigenvectors=True, turbo="on", range=None,
            type=1, overwrite=False):
        """
        Alternative routine to solve the SFA eigenvalue issue. This can be used
        in case the normal symeig() call in _stop_training() throws the common
        SymeigException ('Covariance matrices may be singular').

        It applies PCA to the covariance matrix and filters out rank deficit
        before it applies symeig() to the differential covariance matrix.
        This procedure detects and resolves rank deficit of the covariance
        matrix properly.
        It is roughly twice as expensive as the ordinary SFA implementation
        based on plain symeig.

        Note:
        The advantage compared to prepending a PCA node is that in execution
        phase all data needs to be processed by one step less. That is because
        this approach includes the PCA into the SFA execution matrix.
        """
        if type != 1:
            raise ValueError('Only type=1 is supported.')
        dcov_mtx = A
        cov_mtx = B
        eg, ev = self._symeig(cov_mtx, None, True, turbo, None, type, overwrite)
        off = 0
        while eg[off] < self.rank_threshold:
            off += 1
        self.rank_deficit = off
        eg = 1/numx.sqrt(eg[off:])
        ev2 = ev[:, off:]
        ev2 *= eg
        S = ev2

        white = mult(S.T, mult(dcov_mtx, S))
        sfa_eg, sfa_ev = self._symeig(white, None, True, turbo, None, type, overwrite)
        sfa_ev = mult(S, sfa_ev)
        if range is None:
            # self.d, self.sf
            return sfa_eg, sfa_ev
        else:
            # self.d, self.sf
            return sfa_eg[range[0]-1:range[1]], sfa_ev[:, range[0]-1:range[1]]

    def _rank_deficit_solver_svd(
            self, A, B = None, eigenvectors=True, turbo="on", range=None,
            type=1, overwrite=False):
        """
        Alternative routine to solve the SFA eigenvalue issue. This can be used
        in case the normal symeig() call in _stop_training() throws the common
        SymeigException ('Covariance matrices may be singular').

        This solver's computational cost depends on the underlying svd
        implementation. Its dominant cost factor consists of two svd runs.

        For details on the used algorithm see:
            http://www.geo.tuwien.ac.at/downloads/tm/svd.pdf (section 0.3.2)

        Note:
        The parameters eigenvectors, turbo, type, overwrite are not used.
        They only exist to provide a symeig compatible signature.
        """
        dcov_mtx = A
        cov_mtx = B
        U, s, _ = svd(cov_mtx)
        off = 0
        while s[-1-off] < self.rank_threshold:
            off += 1
        if off > 0:
            self.rank_deficit = off
            s = s[:-off]
            U = U[:, :-off]
        X1 = mult(U, numx.diag(1.0 / s ** 0.5))
        X2, _, _ = svd(mult(X1.T, mult(dcov_mtx, X1)))
        E = mult(X1, X2)
        e = mult(E.T, mult(dcov_mtx, E)).diagonal()

        e = e[::-1]      # SVD delivers the eigenvalues sorted in reverse (compared to symeig). Thus
        E = E.T[::-1].T  # we manually reverse the array/matrix storing the eigenvalues/vectors.

        if range is None:
            #d, sf
            return e, E
        else:
            #d, sf
            return e[range[0] - 1:range[1]], E[:, range[0] - 1:range[1]]


class SFA2Node(SFANode):
    """Get an input signal, expand it in the space of
    inhomogeneous polynomials of degree 2 and extract its slowly varying
    components. The ``get_quadratic_form`` method returns the input-output
    function of one of the learned unit as a ``QuadraticForm`` object.
    See the documentation of ``mdp.utils.QuadraticForm`` for additional
    information.

    More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
    Learning of Invariances, Neural Computation, 14(4):715-770 (2002)."""

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 include_last_sample=True):
        self._expnode = mdp.nodes.QuadraticExpansionNode(input_dim=input_dim,
                                                         dtype=dtype)
        super(SFA2Node, self).__init__(input_dim, output_dim, dtype,
                                       include_last_sample)

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
        If 'n' is an integer, then use the first 'n' slowest components."""
        return super(SFA2Node, self)._execute(self._expnode(x), n)

    def get_quadratic_form(self, nr):
        """
        Return the matrix H, the vector f and the constant c of the
        quadratic form 1/2 x'Hx + f'x + c that defines the output
        of the component 'nr' of the SFA node.
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
