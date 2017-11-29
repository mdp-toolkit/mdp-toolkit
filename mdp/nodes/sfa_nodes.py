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
          ldl  - works by ldl decomposition (requires SciPy >= 1.0)

          auto - selects the best-benchmarked method of the above

          Note: If you already received an exception
          SymeigException ('Covariance matrices may be singular')
          you can manually set the solving method for an existing node:

             sfa.set_rank_deficit_method('pca')

          That means,

             sfa = SFANode(rank_deficit='pca')

          is equivalent to

             sfa = SFANode()
             sfa.set_rank_deficit_method('pca')

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
        self.set_rank_deficit_method(rank_deficit_method)
        self.rank_threshold = 1e-12
        self.rank_deficit = 0

        # SFA eigenvalues and eigenvectors, will be set after training
        self.d = None
        self.sf = None  # second index for outputs
        self.avg = None
        self._bias = None  # avg multiplied with sf
        self.tlen = None

    def set_rank_deficit_method(self, rank_deficit_method):
        if rank_deficit_method == 'pca':
            self._sfa_solver = self._rank_deficit_solver_pca
        elif rank_deficit_method == 'reg':
            self._sfa_solver = self._rank_deficit_solver_reg
        elif rank_deficit_method == 'svd':
            self._sfa_solver = self._rank_deficit_solver_svd
        elif rank_deficit_method == 'ldl':
            try:
                from scipy.linalg.lapack import dsytrf
            except ImportError:
                err_msg = ("ldl method for solving SFA with rank deficit covariance "
                           "requires at least SciPy 1.0.")
                raise NodeException(err_msg)
            self._sfa_solver = self._rank_deficit_solver_ldl
        elif rank_deficit_method == 'auto':
            self._sfa_solver = self._rank_deficit_solver_pca
        else:
            self._sfa_solver = None

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
                err_msg = ("Got negative eigenvalues: %s.\n"
                           "You may either set output_dim to be smaller,\n"
                           "or prepend the SFANode with a PCANode(reduce=True)\n"
                           "or PCANode(svd=True)\n"
                           "or %s."% (str(d), "set a rank deficit method, e.g.\n"
                           "create the SFA node with rank_deficit_method='auto'\n"
                           "and try higher values for rank_threshold, e.g. try\n"
                           "your_node.rank_threshold = 1e-10, 1e-8, 1e-6, ..."
                           if self._sfa_solver is None else
                           "set a higher value for rank_threshold, e.g. try\n"
                           "your_node.rank_threshold = 1e-10, 1e-8, 1e-6, ..."))
                raise NodeException(err_msg)
        except SymeigException as exception:
            errstr = (str(exception)+"\n Covariance matrices may be singular."
                    +"\n Try to create SFA node with rank_deficit_method='auto'.")
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
        For full range, this procedure is (approximately) as efficient as the
        ordinary SFA implementation based on plain symeig, because all
        additional steps are computationally cheap.
        For shorter range, the LDL method should be preferred.
        

        Note:
        For efficiency reasons it actually modifies the covariance matrix
        (even if overwrite=False), but the changes are negligible.
        """
        if type != 1:
            raise ValueError('Only type=1 is supported.')

        # apply some regularization...
        # The following is equivalent to B += 1e-12*np.eye(B.shape[0]),
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
            # In this case we search all indices:
            m_idx = (m < 0.5).nonzero()[0]
            eg = eg[m_idx]
            ev = ev[:, m_idx]
        if range is None:
            return eg, ev
        else:
            return eg[range[0]-1:range[1]], ev[:, range[0]-1:range[1]]

    def _find_blank_data_idx(self, B):
        """
        Helper for some of the rank_deficit solvers.
        Some numerical decompositions, e.g. eig, svd, ldl appear to
        yield numerically unstable results, if the input matrix contains
        blank lines and columns (assuming symmetry).
        It is relevant to guard this case, because it corresponds to constants
        in the data. Think of a constantly white corner in training images or
        slight black stripe atop of a training video due to insufficient cropping
        or think of a logo or timestamp in a video. There are plenty of examples
        that cause constants in real-world data. So by checking for this kind of
        issue we release some burden of inconvenient preprocessing from the user.
        """
        zero_idx = (abs(B[0]) < self.rank_threshold).nonzero()[0]
        # For efficiency we first just check the first line for zeros and fail fast.
        if len(zero_idx) > 0:
            # If there are near- zero entries in first line we check the whole columns:
            #nonzerolines = (abs(numx.sum(B, 0)) > self.rank_threshold).nonzero()[0]
            zero_idx = (numx.mean(abs(B[zero_idx]), 0) < self.rank_threshold).nonzero()[0]
            if len(zero_idx) > 0:
                nonzero_idx = numx.arange(len(B))
                nonzero_idx[zero_idx] = -1
                return nonzero_idx[(nonzero_idx != -1).nonzero()[0]]

    def _rank_deficit_solver_ldl(
            self, A, B = None, eigenvectors=True, turbo="on", rng=None,
            type=1, overwrite=False):
        """
        Alternative routine to solve the SFA eigenvalue issue. This can be used
        in case the normal symeig() call in _stop_training() throws the common
        SymeigException ('Covariance matrices may be singular').

        This solver uses SciPy's raw LAPACK interface to access LDL decomposition.
        www.netlib.org/lapack/lug/node54.html describes how to solve a
        generalized eigenvalue problem with positive definite B using a Cholesky/LL
        decomposition. We extend this method to solve for a positive semidefinite B
        using LDL decomposition, which is a variant of Cholesky/LL decomposition
        for indefinite Matrices.
        Accessing raw LAPACK's LDL decomposition (sytrf) is challenging. This code
        is partly based on code for SciPy 1.1:
        github.com/scipy/scipy/pull/7941/files#diff-9bf9b4b2f0f40415bc0e72143584c889
        We optimized and shortened that code for the real-valued positive
        semidefinite case.

        This procedure is almost as efficient as the ordinary SFA implementation
        based on plain symeig.
        This is because implementations for symmetric generalized eigenvalue problems
        usually perform the Cholesky approach mentioned above. The more general LDL
        decomposition is only slightly more expensive than Cholesky, due to
        pivotization.

        Note:
        This method requires SciPy >= 1.0.
        """
        if type != 1:
            raise ValueError('Only type=1 is supported.')

        # LDL-based method appears to be particularly unstable if blank lines
        # and columns exist in B. So we circumvent this case:
        nonzero_idx = self._find_blank_data_idx(B)
        if not nonzero_idx is None:
            orig_shape = B.shape
            B = B[nonzero_idx, :][:, nonzero_idx]
            A = A[nonzero_idx, :][:, nonzero_idx]

        # This method has special requirements, which is why we import here
        # rather than module wide.
        from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork
        from scipy.linalg.blas import get_blas_funcs
        try:
            inv_tri, solver, solver_lwork = get_lapack_funcs(
                    ('trtri', 'sytrf', 'sytrf_lwork'), (B,))
            mult_tri, = get_blas_funcs(('trmm',), (B,))
        except ValueError:
            err_msg = ("ldl method for solving SFA with rank deficit covariance "
                       "requires at least SciPy 1.0.")
            raise NodeException(err_msg)

        n = B.shape[0]
        arng = numx.arange(n)
        lwork = _compute_lwork(solver_lwork, n, lower=1)
        lu, piv, _ = solver(B, lwork=lwork, lower=1, overwrite_a=overwrite)

        # using piv properly requires some postprocessing:
        swap_ = numx.arange(n)
        pivs = numx.zeros(swap_.shape, dtype=int)
        skip_2x2 = False
        for ind in range(n):
            # If previous spin belonged already to a 2x2 block
            if skip_2x2:
                skip_2x2 = False
                continue

            cur_val = piv[ind]
            # do we have a 1x1 block or not?
            if cur_val > 0:
                if cur_val != ind+1:
                    # Index value != array value --> permutation required
                    swap_[ind] = swap_[cur_val-1]
                pivs[ind] = 1
            # Not.
            elif cur_val < 0 and cur_val == piv[ind+1]:
                # first neg entry of 2x2 block identifier
                if -cur_val != ind+2:
                    # Index value != array value --> permutation required
                    swap_[ind+1] = swap_[-cur_val-1]
                pivs[ind] = 2
                skip_2x2 = True

        full_perm = numx.arange(n)
        for ind in range(n-1, -1, -1):
            s_ind = swap_[ind]
            if s_ind != ind:
                col_s = ind if pivs[ind] else ind-1 # 2x2 block
                lu[[s_ind, ind], col_s:] = lu[[ind, s_ind], col_s:]
                full_perm[[s_ind, ind]] = full_perm[[ind, s_ind]]
        # usually only a few indices actually permute, so we reduce perm:
        perm = (full_perm-arng).nonzero()[0]
        perm_idx = full_perm[perm]
        # end of ldl postprocessing
        # perm_idx and perm now describe a permutation as dest and source indexes

        lu[perm_idx, :] = lu[perm, :]

        dgd = abs(numx.diag(lu))
        dnz = (dgd>self.rank_threshold).nonzero()[0]
        dgd_sqrt_I = numx.sqrt(1.0/dgd[dnz])
        self.rank_deficit = len(dgd) - len(dnz)

        # c, lower, unitdiag, overwrite_c
        LI, _ = inv_tri(lu, 1, 1, 1) # invert triangular
        # we mainly apply tril here, because we need to make a
        # copy of LI anyway, because original result from
        # dtrtri seems to be read-only regarding some operations
        LI = numx.tril(LI, -1)
        LI[arng, arng] = 1
        LI[dnz, :] *= dgd_sqrt_I.reshape((dgd_sqrt_I.shape[0], 1))

        A2 = A if overwrite else A.copy()
        A2[perm_idx, :] = A2[perm, :]
        A2[:, perm_idx] = A2[:, perm]
        # alpha, a, b, side 0=left 1=right, lower, trans_a, diag 1=unitdiag, overwrite_b
        A2 = mult_tri(1.0, LI, A2, 1, 1, 1, 0, 1) # A2 = mult(A2, LI.T)
        A2 = mult_tri(1.0, LI, A2, 0, 1, 0, 0, 1) # A2 = mult(LI, A2)
        A2 = A2[dnz, :]
        A2 = A2[:, dnz]

        # overwrite=True is okay here, because at this point A2 is a copy anyway
        eg, ev = self._symeig(A2, None, True, turbo, rng, overwrite=True)
        ev = mult(LI[dnz].T, ev) if self.rank_deficit \
            else mult_tri(1.0, LI, ev, 0, 1, 1, 0, 1)
        ev[perm] = ev[perm_idx]

        if not nonzero_idx is None:
            # restore ev to original size
            self.rank_deficit += orig_shape[0]-len(nonzero_idx)
            ev_tmp = ev
            ev = numx.zeros((orig_shape[0], ev.shape[1]))
            ev[nonzero_idx, :] = ev_tmp

        return eg, ev

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

        # PCA-based method appears to be particularly unstable if blank lines
        # and columns exist in B. So we circumvent this case:
        nonzero_idx = self._find_blank_data_idx(B)
        if not nonzero_idx is None:
            orig_shape = B.shape
            B = B[nonzero_idx, :][:, nonzero_idx]
            A = A[nonzero_idx, :][:, nonzero_idx]

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
        eg, ev = self._symeig(white, None, True, turbo, range, type, overwrite)
        ev = mult(S, ev)

        if not nonzero_idx is None:
            # restore ev to original size
            self.rank_deficit += orig_shape[0]-len(nonzero_idx)
            ev_tmp = ev
            ev = numx.zeros((orig_shape[0], ev.shape[1]))
            ev[nonzero_idx, :] = ev_tmp

        return eg, ev

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
        if type != 1:
            raise ValueError('Only type=1 is supported.')

        # SVD-based method appears to be particularly unstable if blank lines
        # and columns exist in B. So we circumvent this case:
        nonzero_idx = self._find_blank_data_idx(B)
        if not nonzero_idx is None:
            orig_shape = B.shape
            B = B[nonzero_idx, :][:, nonzero_idx]
            A = A[nonzero_idx, :][:, nonzero_idx]

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

        if not range is None:
            e = e[range[0] - 1:range[1]]
            E = E[:, range[0] - 1:range[1]]

        if not nonzero_idx is None:
            # restore ev to original size
            self.rank_deficit += orig_shape[0]-len(nonzero_idx)
            E_tmp = E
            E = numx.zeros((orig_shape[0], E.shape[1]))
            E[nonzero_idx, :] = E_tmp

        return e, E


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
                 include_last_sample=True, rank_deficit_method='none'):
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
