'''
Contains functions to deal with generalized symmetric positive semidefinite
eigenvalue problems Av = lBv where A and B are real symmetric matrices and
B must be positive semidefinite. Note that eigh in SciPy requires positive
definite B, i.e. requires strictly positive eigenvalues in B.
There are plenty of cases where real-world data leads to a covariance
matrix with rank deficit, e.g. a constantly white corner in training images
or a black stripe atop of a training video due to insufficient cropping or
a logo or timestamp in a video.
In such cases nodes like SFA usually fail with
SymeigException ('Covariance matrices may be singular').
The functions in this module allow for a more robust data processing in
such scenarios.

The various functions in this module yield different advantages and
disadvantages in terms of robustness, accuracy, performance and requirements.
Each of them can fail due to ill-conditioned data and if you experience
problems with specific data you can try another method or use a higher
value for rank_threshold, e.g. 1e-10, 1e-8, 1e-6, ... (default: 1e-12).

A short overview (for more details see the doc of each function):

pca (solves by prepending PCA/ principal component analysis)

  One of the most stable and accurate approaches.
  Roughly twice as expensive as ordinary symmetric eigenvalue solving as
  it solves two symmetric eigenvalue problems.
  Only the second one can exploit range parameter for performance.


svd (solves by SVD / singular value decomposition)

  One of the most stable and accurate approaches.
  Involves solving two svd problems. Computational cost can vary greatly
  depending on the backends used. E.g. SVD from SciPy appears to be much
  faster than SVD from NumPy. Based on this it can be faster or slower
  than the PCA based approach.


reg (solves by applying regularization to matrix B)

  Roughly as efficient as ordinary eigenvalue solving if no range is given.
  If range is given, depending on the backend for ordinary symmetric
  eigenvalue solving, this method can be much slower than an ordinary
  symmetric eigenvalue solver that can exploit range for performance.


ldl (solves by applying LDL / Cholesky decomposition for indefinite matrices)

  Roughly as efficient as ordinary eigenvalue solving. Can exploit range
  parameter for performance just as well as the backend for ordinary symmetric
  eigenvalue solving enables. This is the recommended and most efficient
  approach, but it requires SciPy 1.0 or newer.
'''

import mdp
from mdp import numx
from ._symeig import SymeigException


def symeig_semidefinite_reg(
        A, B = None, eigenvectors=True, turbo="on", range=None,
        type=1, overwrite=False, rank_threshold=1e-12, dfc_out=None):
    """
    Regularization-based routine to solve generalized symmetric positive
    semidefinite eigenvalue problems.
    This can be used in case the normal symeig() call in _stop_training()
    throws SymeigException ('Covariance matrices may be singular').

    This solver applies a moderate regularization to B before applying
    eigh/symeig. Afterwards it properly detects the rank deficit and
    filters out malformed features.
    For full range, this procedure is (approximately) as efficient as the
    ordinary eigh implementation, because all additional steps are
    computationally cheap.
    For shorter range, the LDL method should be preferred.


    The signature of this function equals that of mdp.utils.symeig, but
    has two additional parameters:
    
    rank_threshold: A threshold to determine if an eigenvalue counts as zero.
    
    dfc_out: If dfc_out is not None dfc_out.rank_deficit will be set to an
             integer indicating how many zero-eigenvalues were detected.


    Note:
    For efficiency reasons it actually modifies the matrix B
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
    Bflat[idx] += rank_threshold

    eg, ev = mdp.utils.symeig(A, B, True, turbo, None, type, overwrite)

    Bflat[idx] = diag_tmp
    m = numx.absolute(numx.sqrt(numx.absolute(
            numx.sum(ev * mdp.utils.mult(B, ev), 0)))-1)
    off = 0
    # In theory all values in m should be close to one or close to zero.
    # So we use the mean of these values as threshold to distinguish cases:
    while m[off] > 0.5:
        off += 1
    m_off_sum = numx.sum(m[off:])
    if m_off_sum < 0.5:
        if off > 0:
            if not dfc_out is None:
                dfc_out.rank_deficit = off
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


def _find_blank_data_idx(B, rank_threshold):
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
    zero_idx = (abs(B[0]) < rank_threshold).nonzero()[0]
    # For efficiency we just check the first line for zeros and fail fast.
    if len(zero_idx) > 0:
        # If near-zero entries are in first line we check the whole columns:
        #nonzerolines = (abs(numx.sum(B, 0)) > rank_threshold).nonzero()[0]
        zero_idx = (numx.mean(abs(B[zero_idx]), 0) < \
                rank_threshold).nonzero()[0]
        if len(zero_idx) > 0:
            nonzero_idx = numx.arange(len(B))
            nonzero_idx[zero_idx] = -1
            return nonzero_idx[(nonzero_idx != -1).nonzero()[0]]


def symeig_semidefinite_ldl(
        A, B = None, eigenvectors=True, turbo="on", rng=None,
        type=1, overwrite=False, rank_threshold=1e-12, dfc_out=None):
    """
    LDL-based routine to solve generalized symmetric positive semidefinite
    eigenvalue problems.
    This can be used in case the normal symeig() call in _stop_training()
    throws SymeigException ('Covariance matrices may be singular').

    This solver uses SciPy's raw LAPACK interface to access LDL decomposition.
    www.netlib.org/lapack/lug/node54.html describes how to solve a
    generalized eigenvalue problem with positive definite B using Cholesky/LL
    decomposition. We extend this method to solve for positive semidefinite B
    using LDL decomposition, which is a variant of Cholesky/LL decomposition
    for indefinite Matrices.
    Accessing raw LAPACK's LDL decomposition (sytrf) is challenging. This code
    is partly based on code for SciPy 1.1:
    github.com/scipy/scipy/pull/7941/files#diff-9bf9b4b2f0f40415bc0e72143584c889
    We optimized and shortened that code for the real-valued positive
    semidefinite case.

    This procedure is almost as efficient as the ordinary eigh implementation.
    This is because implementations for symmetric generalized eigenvalue
    problems usually perform the Cholesky approach mentioned above. The more
    general LDL decomposition is only slightly more expensive than Cholesky,
    due to pivotization.


    The signature of this function equals that of mdp.utils.symeig, but
    has two additional parameters:
    
    rank_threshold: A threshold to determine if an eigenvalue counts as zero.
    
    dfc_out: If dfc_out is not None dfc_out.rank_deficit will be set to an
             integer indicating how many zero-eigenvalues were detected.


    Note:
    This method requires SciPy >= 1.0.
    """
    if type != 1:
        raise ValueError('Only type=1 is supported.')

    # LDL-based method appears to be particularly unstable if blank lines
    # and columns exist in B. So we circumvent this case:
    nonzero_idx = _find_blank_data_idx(B, rank_threshold)
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
        err_msg = ("ldl method for solving symeig with rank deficit B "
                   "requires at least SciPy 1.0.")
        raise SymeigException(err_msg)

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
    dnz = (dgd > rank_threshold).nonzero()[0]
    dgd_sqrt_I = numx.sqrt(1.0/dgd[dnz])
    rank_deficit = len(dgd) - len(dnz) # later used

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
    # alpha, a, b, side 0=left 1=right, lower, trans_a, diag 1=unitdiag,
    # overwrite_b
    A2 = mult_tri(1.0, LI, A2, 1, 1, 1, 0, 1) # A2 = mult(A2, LI.T)
    A2 = mult_tri(1.0, LI, A2, 0, 1, 0, 0, 1) # A2 = mult(LI, A2)
    A2 = A2[dnz, :]
    A2 = A2[:, dnz]

    # overwrite=True is okay here, because at this point A2 is a copy anyway
    eg, ev = mdp.utils.symeig(A2, None, True, turbo, rng, overwrite=True)
    ev = mdp.utils.mult(LI[dnz].T, ev) if rank_deficit \
        else mult_tri(1.0, LI, ev, 0, 1, 1, 0, 1)
    ev[perm] = ev[perm_idx]

    if not nonzero_idx is None:
        # restore ev to original size
        rank_deficit += orig_shape[0]-len(nonzero_idx)
        ev_tmp = ev
        ev = numx.zeros((orig_shape[0], ev.shape[1]))
        ev[nonzero_idx, :] = ev_tmp

    if not dfc_out is None:
        dfc_out.rank_deficit = rank_deficit
    return eg, ev


def symeig_semidefinite_pca(
        A, B = None, eigenvectors=True, turbo="on", range=None,
        type=1, overwrite=False, rank_threshold=1e-12, dfc_out=None):
    """
    PCA-based routine to solve generalized symmetric positive semidefinite
    eigenvalue problems.
    This can be used in case the normal symeig() call in _stop_training()
    throws SymeigException ('Covariance matrices may be singular').

    It applies PCA to B and filters out rank deficit before it applies
    symeig() to A.
    It is roughly twice as expensive as the ordinary eigh implementation.


    The signature of this function equals that of mdp.utils.symeig, but
    has two additional parameters:
    
    rank_threshold: A threshold to determine if an eigenvalue counts as zero.
    
    dfc_out: If dfc_out is not None dfc_out.rank_deficit will be set to an
             integer indicating how many zero-eigenvalues were detected.


    Note:
    The advantage compared to prepending a PCA node is that in execution
    phase all data needs to be processed by one step less. That is because
    this approach includes the PCA into e.g. the SFA execution matrix.
    """
    if type != 1:
        raise ValueError('Only type=1 is supported.')
    mult = mdp.utils.mult

    # PCA-based method appears to be particularly unstable if blank lines
    # and columns exist in B. So we circumvent this case:
    nonzero_idx = _find_blank_data_idx(B, rank_threshold)
    if not nonzero_idx is None:
        orig_shape = B.shape
        B = B[nonzero_idx, :][:, nonzero_idx]
        A = A[nonzero_idx, :][:, nonzero_idx]

    dcov_mtx = A
    cov_mtx = B
    eg, ev = mdp.utils.symeig(cov_mtx, None, True, turbo, None, type,
			overwrite)
    off = 0
    while eg[off] < rank_threshold:
        off += 1
    if not dfc_out is None:
        dfc_out.rank_deficit = off
    eg = 1/numx.sqrt(eg[off:])
    ev2 = ev[:, off:]
    ev2 *= eg
    S = ev2

    white = mult(S.T, mult(dcov_mtx, S))
    eg, ev = mdp.utils.symeig(white, None, True, turbo, range, type,
			overwrite)
    ev = mult(S, ev)

    if not nonzero_idx is None:
        # restore ev to original size
        if not dfc_out is None:
            dfc_out.rank_deficit += orig_shape[0]-len(nonzero_idx)
        ev_tmp = ev
        ev = numx.zeros((orig_shape[0], ev.shape[1]))
        ev[nonzero_idx, :] = ev_tmp

    return eg, ev


def symeig_semidefinite_svd(
        A, B = None, eigenvectors=True, turbo="on", range=None,
        type=1, overwrite=False, rank_threshold=1e-12, dfc_out=None):
    """
    SVD-based routine to solve generalized symmetric positive semidefinite
    eigenvalue problems.
    This can be used in case the normal symeig() call in _stop_training()
    throws SymeigException ('Covariance matrices may be singular').

    This solver's computational cost depends on the underlying SVD
    implementation. Its dominant cost factor consists of two SVD runs.
    
    rank_threshold=1e-12
    dfc_out=None

    For details on the used algorithm see:
        http://www.geo.tuwien.ac.at/downloads/tm/svd.pdf (section 0.3.2)


    The signature of this function equals that of mdp.utils.symeig, but
    has two additional parameters:
    
    rank_threshold: A threshold to determine if an eigenvalue counts as zero.
    
    dfc_out: If dfc_out is not None dfc_out.rank_deficit will be set to an
             integer indicating how many zero-eigenvalues were detected.


    Note:
    The parameters eigenvectors, turbo, type, overwrite are not used.
    They only exist to provide a symeig compatible signature.
    """
    if type != 1:
        raise ValueError('Only type=1 is supported.')
    mult = mdp.utils.mult

    # SVD-based method appears to be particularly unstable if blank lines
    # and columns exist in B. So we circumvent this case:
    nonzero_idx = _find_blank_data_idx(B, rank_threshold)
    if not nonzero_idx is None:
        orig_shape = B.shape
        B = B[nonzero_idx, :][:, nonzero_idx]
        A = A[nonzero_idx, :][:, nonzero_idx]

    dcov_mtx = A
    cov_mtx = B
    U, s, _ = mdp.utils.svd(cov_mtx)
    off = 0
    while s[-1-off] < rank_threshold:
        off += 1
    if off > 0:
        if not dfc_out is None:
            dfc_out.rank_deficit = off
        s = s[:-off]
        U = U[:, :-off]
    X1 = mult(U, numx.diag(1.0 / s ** 0.5))
    X2, _, _ = mdp.utils.svd(mult(X1.T, mult(dcov_mtx, X1)))
    E = mult(X1, X2)
    e = mult(E.T, mult(dcov_mtx, E)).diagonal()

    e = e[::-1]      # SVD delivers the eigenvalues sorted in reverse (compared to symeig). Thus
    E = E.T[::-1].T  # we manually reverse the array/matrix storing the eigenvalues/vectors.

    if not range is None:
        e = e[range[0] - 1:range[1]]
        E = E[:, range[0] - 1:range[1]]

    if not nonzero_idx is None:
        # restore ev to original size
        if not dfc_out is None:
            dfc_out.rank_deficit += orig_shape[0]-len(nonzero_idx)
        E_tmp = E
        E = numx.zeros((orig_shape[0], E.shape[1]))
        E[nonzero_idx, :] = E_tmp

    return e, E
