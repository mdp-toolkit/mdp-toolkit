
#====================================================================[ header ]

from builtins import range
__docformat__ = "restructuredtext en"

import numpy as np
import matplotlib.pyplot as plt

import mdp
from mdp import Node, NodeException, TrainingException
from mdp.utils import (mult, pinv, CovarianceMatrix, QuadraticForm,
                       symeig, SymeigException)

#----------------------------------------------------------------------[ util ]

def generateID():  # unique id generator (see _data_analysis)
	generateID.i = 0 if 'i' not in generateID.__dict__ else generateID.i+1
	return generateID.i

#===================================================================[ SFANode ]

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
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 include_last_sample=True, use_svd=False, always_analyse_data=False):
        """
        Extract the slowly varying components from the input data.

        Args:
            input_dim  (int): Dimensionality of the data going into the node.
            output_dim (int): Dimensionality of the data leaving the node.
            dtype     (type): Type information for the data (e.g. numpy.float64).
            use_svd   (bool): Use SVD instead of the default symeig procedure to
                              solve the SFA-internal eigenvalue problem.
            include_last_sample: See SFANode docstring or API doc [1].
            always_analyse_data: Produce additional information on the training
                              data (always or only when training fails).

        [1] http://mdp-toolkit.sourceforge.net/api/mdp.nodes.SFANode-class.html
         """
        super(SFANode, self).__init__(input_dim, output_dim, dtype)
        self._include_last_sample = include_last_sample

        # init two covariance matrices
        # one for the input data
        self._cov_mtx  = CovarianceMatrix(dtype)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(dtype)

        # set parameters for solving the eigenproblem
        self._symeig   = symeig
        self._use_svd  = use_svd
        self._analysis = always_analyse_data

        # SFA eigenvalues and eigenvectors, will be set after training
        self.d     = None
        self.sf    = None  # second index for outputs
        self.avg   = None
        self._bias = None  # avg multiplied with sf
        self.tlen  = None

    def _set_range(self):
        if self.output_dim is not None and self.output_dim <= self.input_dim:
            # (eigenvalues sorted in ascending order)
            rng = (1, self.output_dim)
        else:
            # otherwise, keep all output components
            rng = None
            self.output_dim = self.input_dim
        return rng

    def _inverse(self, y):
        return mult(y, pinv(self.sf)) + self.avg

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative."""
        # this is faster than a linear_filter or a weave-inline solution
        return x[1:, :]-x[:-1, :]

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
        return self._refcast(t / (2 * np.pi) * np.sqrt(self.d))

    #--------------------------------------------------------------[ training ]

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

        # request the covariance matrices and clean up
        self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx

        # do not center around the mean:
        # we want the second moment matrix (centered about 0) and
        # not the second central moment matrix (centered about the mean), i.e.
        # the covariance matrix
        self.dcov_mtx, self.davg, self.dtlen = self._dcov_mtx.fix(center=False)
        del self._dcov_mtx

        rng = self._set_range()

        ####################################################################### EIG

        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        if not self._use_svd:
            try:
                self.d, self.sf = self._symeig(self.dcov_mtx, self.cov_mtx,
                                         range=rng, overwrite=(not debug))
                d = self.d
                # check that we get only *positive* eigenvalues
                if d.min() < 0:
                    err_msg = ("Got negative eigenvalues: %s."
                               " You may either set output_dim to be smaller,"
                               " or prepend the SFANode with a PCANode(reduce=True)"
                               " or PCANode(svd=True)"% str(d))
                    raise NodeException(err_msg)

            except SymeigException as exception:
                errstr  = str(exception)+"\n Covariance matrices may be singular."
                errstr += self._data_fail_msg()  # offer some additional information on this issue
                self._data_analysis()            # and print some information about the training set
                raise NodeException(errstr)

        else:
            self._svd_solver( rng )  # sets self.d and self.sf

        ####################################################################### EIG

        if not debug:
            # delete covariance matrix if no exception occurred
            del self.cov_mtx
            del self.dcov_mtx

        # store bias
        self._bias = mult(self.avg, self.sf)

        # plot some statistics of the training data (optional)
        if self._analysis: self._data_analysis()

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

    def _svd_solver(self, rng):
        """
        Alternative routine to solve the SFA eigenvalue issue. This can be used
        in case the normal symeig() call in _stop_training() throws the common
        SymeigException ('Covariance matrices may be singular').

        For details on the used algorithm see:
            http://www.geo.tuwien.ac.at/downloads/tm/svd.pdf (section 0.3.2)
        """

        U, s, _ = np.linalg.svd(self.cov_mtx)
        X1 = np.dot(U, np.diag(1.0 / s ** 0.5))
        X2, _, _ = np.linalg.svd(np.dot(X1.T, np.dot(self.dcov_mtx, X1)))
        E = np.dot(X1, X2)
        e = np.dot(E.T, np.dot(self.dcov_mtx, E)).diagonal()

        e = e[::-1]      # SVD delivers the eigenvalues sorted in reverse (compared to symeig). Thus
        E = E.T[::-1].T  # we manually reverse the array/matrix storing the eigenvalues/vectors.

        if rng is None:
            self.d, self.sf = e, E
        else:
            self.d, self.sf = e[rng[0] - 1:rng[1]], E[:, rng[0] - 1:rng[1]]

    #-------------------------------------------------------------[ debugging ]

    def _data_fail_msg(self):
        """
        This methods provides additional information in case _stop_training()
        runs into a singular matrix and cannot continue. Since this is a common
        problem, this message expands on the issue and offers different ways to
        solve it.

        Returns:
            string: Additional string to be appended to the default error.
        """
        return "This usually happens if there are redundancies in the (expanded) training data.\n" + \
               " There are several ways to deal with this issue:\n\n" + \
               "   - Use more data.\n\n" + \
               "   - Add noise to the data. This can be done by slotting an additional NoiseNode\n" + \
               "     in front of a troublesome SFANode. Noise levels do not have to be high.\n\n" + \
               "   - Run training data through PCA. This can be done by slotting an additional\n" + \
               "     PCA node in front of the troublesome SFANode. Use the PCA node to discard\n" + \
               "     dimensions within your data with lower variance.\n\n" + \
               "   - [ RECOMMENDED ] Use SVD by setting the SFANode flag use_svd to True. This\n" + \
               "     solves the internal SFA eigenvalue problem by using two instances of singular\n" + \
               "     value decomposition (SVD) instead of the default behavior of using the\n" + \
               "     systems's symeig() routine to solve a single, generalized eigenvalue problem.\n" + \
               "     Using SVD is slightly more expensive but more robust.\n"

    def _data_analysis(self):
        """
        Function to provide basic data analysis of the used training data. It
        creates a plot depicting the variance and slowness per signal, as well
        as the eigenvalue spectrum of the training data.
        """

        plt.clf()

        self._data_analysis_plotter(np.diagonal(self.cov_mtx),  (3, 2, 1), 'Variance per dimension')
        self._data_analysis_plotter(np.diagonal(self.cov_mtx),  (3, 2, 2), 'Dimensions by Variance', arg_sort=True)
        self._data_analysis_plotter(np.diagonal(self.dcov_mtx), (3, 2, 3), 'Slowness per dimension')
        self._data_analysis_plotter(np.diagonal(self.dcov_mtx), (3, 2, 4), 'Dimensions by slowness', arg_sort=True)

        _, s, _ = np.linalg.svd( self.cov_mtx )
        self._data_analysis_plotter(s, (3, 2, 5), 'Eigenvalue spectrum')

        plt.tight_layout()
        plt.savefig( 'SFANode['+str(generateID())+'] training data analysis.png' )

    def _data_analysis_plotter(self, x, sub_pos, title='', arg_sort=False, style='dots'):
        """
        Helper function for _data_analysis(). Here the actual (sub) plots are
        being generated.

        Args:
            x (numerical array): 1D array of data to be plotted.
            sub_pos ((int,int,int) tupel): Position of the sub plot in the overall plot.
            arg_sort (bool): Sort by dimension (0..n) [False] or by value per dimension [True].
            style ('dots'/'bars'): Plot data as set of 'dots' or 'bars'.
        """
        # plot and axis setup
        ax = plt.subplot( sub_pos[0], sub_pos[1], sub_pos[2] )
        if title is not '': ax.set_title( title )
        x_rng    = np.arange( len(x) )
        x_labels = x_rng
        # sort data by dimension (default) or value (arg_sort)
        if arg_sort:
            sort = np.argsort( x )
            x_labels = x_labels[sort]
            x = x[sort]
        # plot
        if style is 'bars':
            plt.bar( x_rng, x, align='center' )
        elif style is 'dots':
            plt.plot( x, 'o' )
        # ticks and tick labels
        if len(x) <= 50: plt.xticks( x_rng )
        ax.set_xticklabels( x_labels )
        ax.set_xlim( (-1,len(x_rng)) )

#==================================================================[ SFA2Node ]

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
        h = np.zeros((n, n), dtype=self.dtype)
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
