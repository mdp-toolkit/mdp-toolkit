###############################################################################
# gsfa_nodes: This module implements the Graph-Based SFA Node (GSFANode) and  #
# the Information-Preserving GSFA Node (iGSFANode).                           #
#                                                                             #
# See the following publications for details on GSFA and iGSFA:               #
# * Escalante-B A.-N., Wiskott L, "How to solve classification and regression #
# problems on high-dimensional data with a supervised extension of Slow       #
# Feature Analysis". Journal of Machine Learning Research 14:3683-3719, 2013  #
# * Escalante-B., A.-N. and Wiskott, L., "Improved graph-based {SFA}:         #
# Information preservation complements the slowness principle", e-print       #
# arXiv:1601.03945, http://arxiv.org/abs/1601.03945, 2017                     #
#                                                                             #
# Examples of using GSFA and iGSFA are provided at the end of the file        #
###############################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import mdp
from mdp import numx, NodeException, TrainingException
from mdp.utils import (mult, symeig, pinv, SymeigException)
from mdp.nodes import GeneralExpansionNode

# For Python 2 & 3 compatibility
try:
    basestring
except NameError:
    basestring = str


class GSFANode(mdp.Node):
    """ This node implements "Graph-Based SFA (GSFA)", which is the main
    component of hierarchical GSFA (HGSFA).

    For further information, see: Escalante-B A.-N., Wiskott L, "How to solve
    classification and regression problems on high-dimensional data with a
    supervised extension of Slow Feature Analysis". Journal of Machine
    Learning Research 14:3683-3719, 2013
    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 block_size=None, train_mode=None, verbose=False):
        """Initializes the GSFA node, which is a subclass of the SFA node.

        The parameters block_size and train_mode are not necessary and it is
        recommended to skip them here and provide them as parameters to the
        train method.
        See the _train method for details.
        """
        super(GSFANode, self).__init__(input_dim, output_dim, dtype)
        self.pinv = None
        self.block_size = block_size
        self.train_mode = train_mode
        self.verbose = verbose
        self._symeig = symeig
        self._covdcovmtx = CovDCovMatrix()

        # SFA eigenvalues and eigenvectors, will be set after training
        self.d = None
        self.sf = None  # second index for outputs
        self.avg = None
        self._bias = None  # avg multiplied with sf

    def _train(self, x, block_size=None, train_mode=None, node_weights=None,
               edge_weights=None, verbose=None):
        """This is the main training function of GSFA.

        x: training data (each sample is a row)

        The semantics of the remaining parameters depends on the training mode
        (train_mode) parameter in order to train as in standard SFA:
            set train_mode="regular" (the scale of the features should be
            corrected afterwards)
        in order to train using the clustered graph:
            set train_mode="clustered". The cluster size is given by block_size
            (integer). Variable cluster sizes are possible if block_size is a
            list of integers. Samples belonging to the same class should be
            adjacent.
        in order to train for classification:
            set train_mode=("classification", labels, weight), where labels is
             an array with the class information and weight is a scalar value
             (e.g., 1.0).
        in order to train for regression:
            set train_mode=("serial_regression#", labels, weight), where # is
            an integer that specifies the block size used by a serial graph,
            labels is an array with the label information and weight is a
            scalar value.
        in order to train using a graph without edges:
            set train_mode="unlabeled".
        in order to train using the serial graph:
            set train_mode="serial", and use block_size (integer) to specify
            the group size.
        in order to train using the mixed graph:
            set train_mode="mixed", and use block_size (integer) to specify
            the group size.
        in order to train using an arbitrary user-provided graph:
            set train_mode="graph", specify the node_weights (numpy 1D array),
            and the edge_weights (numpy 2D array).
        """
        if train_mode is None:
            train_mode = self.train_mode
        if verbose is None:
            verbose = self.verbose
        if block_size is None:
            if verbose:
                print("parameter block_size was not provided ",
                      "using default value self.block_size")
            block_size = self.block_size

        self.set_input_dim(x.shape[1])

        if verbose:
            print("train_mode=", train_mode)

        if isinstance(train_mode, list):
            train_modes = train_mode
        else:
            train_modes = [train_mode]

        for train_mode in train_modes:
            if train_mode is None:
                train_mode = "regular"
            if isinstance(train_mode, tuple):
                method = train_mode[0]
                labels = train_mode[1]
                weight = train_mode[2]
                if method == "classification":
                    if verbose:
                        print("update classification")
                    ordering = numx.argsort(labels)
                    x2 = x[ordering, :]
                    unique_labels = numx.unique(labels)
                    unique_labels.sort()
                    block_sizes = []
                    for label in unique_labels:
                        block_sizes.append((labels == label).sum())
                    self._covdcovmtx.update_clustered(x2,
                                                      block_sizes=block_sizes,
                                                      weight=weight)
                elif method.startswith("serial_regression"):
                    block_size = int(method[len("serial_regression"):])
                    if verbose:
                        print("update serial_regression, block_size=",
                              block_size)
                    ordering = numx.argsort(labels)
                    x2 = x[ordering, :]
                    self._covdcovmtx.update_serial(x2, block_size=block_size,
                                                   weight=weight)
                else:
                    er = "method unknown: %s" % (str(method))
                    raise ValueError(er)
            elif isinstance(train_mode, basestring):
                if train_mode == 'unlabeled':
                    if verbose:
                        print("update_unlabeled")
                    self._covdcovmtx.update_unlabeled(x, weight=1.0)
                elif train_mode == "regular":
                    if verbose:
                        print("update_regular")
                    self._covdcovmtx.update_regular(x, weight=1.0)
                elif train_mode == 'clustered':
                    if verbose:
                        print("update_clustered")
                    self._covdcovmtx.update_clustered(x,
                                                      block_sizes=block_size,
                                                      weight=1.0)
                elif train_mode.startswith('compact_classes'):
                    if verbose:
                        print("update_compact_classes:", train_mode)
                    J = int(train_mode[len('compact_classes'):])
                    self._covdcovmtx.update_compact_classes(
                        x, block_sizes=block_size, Jdes=J)
                elif train_mode == 'serial':
                    if verbose:
                        print("update_serial")
                    self._covdcovmtx.update_serial(x, block_size=block_size)
                elif train_mode.startswith('DualSerial'):
                    if verbose:
                        print("updateDualSerial")
                    num_blocks = len(x) // block_size
                    dual_num_blocks = int(train_mode[len("DualSerial"):])
                    dual_block_size = len(x) // dual_num_blocks
                    chunk_size = block_size // dual_num_blocks
                    if verbose:
                        print("dual_num_blocks = ", dual_num_blocks)
                    self._covdcovmtx.update_serial(x, block_size=block_size)
                    x2 = numx.zeros_like(x)
                    for i in range(num_blocks):
                        for j in range(dual_num_blocks):
                            x2[j * dual_block_size + i * chunk_size:
                               j * dual_block_size + (i + 1) * chunk_size] = \
                               x[i * block_size + j * chunk_size:
                                 i * block_size + (j + 1) * chunk_size]
                    self._covdcovmtx.update_serial(x2,
                                                   block_size=dual_block_size,
                                                   weight=0.0)
                elif train_mode == 'mixed':
                    if verbose:
                        print("update mixed")
                    bs = block_size
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(
                        x[0:bs], weight=2.0, block_size=block_size)
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(
                        x[bs:-bs], weight=1.0, block_size=block_size)
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(
                        x[-bs:], weight=2.0, block_size=block_size)
                    self._covdcovmtx.update_serial(x, block_size=block_size)
                elif train_mode[0:6] == 'window':
                    window_halfwidth = int(train_mode[6:])
                    if verbose:
                        print("Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update_sliding_window(
                        x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode[0:7] == 'fwindow':
                    window_halfwidth = int(train_mode[7:])
                    if verbose:
                        print("Fast Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update_fast_sliding_window(
                        x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode[0:13] == 'mirror_window':
                    window_halfwidth = int(train_mode[13:])
                    if verbose:
                        print("Mirroring Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update_mirroring_sliding_window(
                        x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode[0:14] == 'smirror_window':
                    window_halfwidth = int(train_mode[14:])
                    if verbose:
                        print("Slow Mirroring Window (%d)" % window_halfwidth)
                    self._covdcovmtx.update_slow_mirroring_sliding_window(
                        x, weight=1.0, window_halfwidth=window_halfwidth)
                elif train_mode == 'graph':
                    if verbose:
                        print("update_graph")
                    self._covdcovmtx.update_graph(
                        x, node_weights=node_weights,
                        edge_weights=edge_weights, weight=1.0)
                elif train_mode == 'graph_old':
                    if verbose:
                        print("update_graph_old")
                    self._covdcovmtx.update_graph_old(
                        x, node_weights=node_weights,
                        edge_weights=edge_weights, weight=1.0)
                elif train_mode == 'smart_unlabeled2':
                    if verbose:
                        print("smart_unlabeled2")
                    N2 = x.shape[0]

                    N1 = Q1 = self._covdcovmtx.num_samples * 1.0
                    R1 = self._covdcovmtx.num_diffs * 1.0
                    sum_x_labeled_2D = self._covdcovmtx.sum_x.reshape((1, -1))
                    sum_prod_x_labeled = self._covdcovmtx.sum_prod_x
                    if verbose:
                        print("Original sum_x[0]/num_samples=",
                              self._covdcovmtx.sum_x[0] /
                              self._covdcovmtx.num_samples)

                    weight_fraction_unlabeled = 0.2
                    additional_weight_unlabeled = -0.025

                    w1 = Q1 * 1.0 / R1 * (1.0 - weight_fraction_unlabeled)
                    if verbose:
                        print("weight_fraction_unlabeled=",
                              weight_fraction_unlabeled)
                        print("N1=Q1=", Q1, "R1=", R1, "w1=", w1)
                        print("")

                    self._covdcovmtx.sum_prod_diffs *= w1
                    self._covdcovmtx.num_diffs *= w1
                    if verbose:
                        print("After diff scaling: num_samples=",
                              self._covdcovmtx.num_samples)
                        print("num_diffs=", self._covdcovmtx.num_diffs, "\n")

                    node_weights2 = Q1 * weight_fraction_unlabeled / N2
                    w12 = node_weights2 / N1  # One directional weights
                    if verbose:
                        print("w12 (one dir)", w12)

                    sum_x_unlabeled_2D = x.sum(axis=0).reshape((1, -1))
                    sum_prod_x_unlabeled = mdp.utils.mult(x.T, x)

                    self._covdcovmtx.add_samples(sum_prod_x_unlabeled,
                                                 sum_x_unlabeled_2D.flatten(),
                                                 num_samples=N2,
                                                 weight=node_weights2)
                    if verbose:
                        print("After adding unlabeled nodes: num_samples=",
                              self._covdcovmtx.num_samples)
                        print("num_diffs=", self._covdcovmtx.num_diffs)
                        print("sum_x[0]/num_samples=",
                              self._covdcovmtx.sum_x[0] /
                              self._covdcovmtx.num_samples)
                        print("")

                        print("N2=", N2, "node_weights2=", node_weights2)

                    additional_diffs = sum_prod_x_unlabeled * N1 - \
                                       mdp.utils.mult(sum_x_labeled_2D.T,
                                                      sum_x_unlabeled_2D) - \
                                       mdp.utils.mult(sum_x_unlabeled_2D.T,
                                                      sum_x_labeled_2D) + \
                                       sum_prod_x_labeled * N2
                    if verbose:
                        print("w12=", w12, "additional_diffs=",
                              additional_diffs)
                    # accounts for both directions
                    self._covdcovmtx.add_diffs(2 * additional_diffs,
                                               2 * N1 * N2,
                                               weight=w12)
                    if verbose:
                        print("After mixed diff addition: num_samples=",
                              self._covdcovmtx.num_samples)
                        print("num_diffs=", self._covdcovmtx.num_diffs)
                        print("sum_x[0]/num_samples=",
                              self._covdcovmtx.sum_x[0] / \
                              self._covdcovmtx.num_samples)

                        print("\n Adding complete graph for unlabeled data")
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(
                        x, weight=additional_weight_unlabeled, block_size=N2)
                    if verbose:
                        print("After complete x2 addition: num_samples=",
                              self._covdcovmtx.num_samples)
                        print("num_diffs=", self._covdcovmtx.num_diffs)
                        print("sum_x[0]/num_samples=",
                              self._covdcovmtx.sum_x[0] /
                              self._covdcovmtx.num_samples)

                elif train_mode == 'smart_unlabeled3':
                    if verbose:
                        print("smart_unlabeled3")
                    N2 = x.shape[0]

                    N1 = Q1 = self._covdcovmtx.num_samples * 1.0
                    R1 = self._covdcovmtx.num_diffs * 1.0
                    if verbose:
                        print("N1=Q1=", Q1, "R1=", R1, "N2=", N2)

                    v = 2.0 ** (-9.5)  # weight of unlabeled samples
                    C = 10.0  # Clustered graph assumed, with C classes,
                    # and each class having N1/C samples
                    if verbose:
                        print("v=", v, "C=", C)

                    v_norm = v / C
                    N1_norm = N1 / C

                    sum_x_labeled = self._covdcovmtx.sum_x.reshape((1, -1))
                    sum_prod_x_labeled = self._covdcovmtx.sum_prod_x + 0.0

                    if verbose:
                        print("Original (Diag(C')/num_diffs.avg)**0.5 =",
                              ((numx.diagonal(self._covdcovmtx.sum_prod_diffs)
                                / self._covdcovmtx.num_diffs).mean()) ** 0.5)

                    weight_adjustment = (N1_norm - 1) / \
                                        (N1_norm - 1 + v_norm * N2)
                    if verbose:
                        print("weight_adjustment =", weight_adjustment,
                              "w11=", 1 / (N1_norm - 1 + v_norm * N2))

                    self._covdcovmtx.sum_x *= weight_adjustment
                    self._covdcovmtx.sum_prod_x *= weight_adjustment
                    self._covdcovmtx.num_samples *= weight_adjustment
                    self._covdcovmtx.sum_prod_diffs *= weight_adjustment
                    self._covdcovmtx.num_diffs *= weight_adjustment
                    node_weights_complete_1 = weight_adjustment
                    if verbose:
                        print("num_diffs (w11) after weight_adjustment=",
                              self._covdcovmtx.num_diffs)
                    w11 = 1 / (N1_norm - 1 + v_norm * N2)
                    if verbose:
                        print("Adjusted (Diag(C')/num_diffs.avg)**0.5 =",
                              ((numx.diagonal(self._covdcovmtx.sum_prod_diffs)
                                / self._covdcovmtx.num_diffs).mean()) ** 0.5)
                        print("")

                    # Connections within unlabeled data (notice that C times
                    # this is equivalent to v*v/(N1+v*(N2-1)) once)
                    w22 = 0.5 * 2 * v_norm * v_norm / (N1_norm +
                                                       v_norm * (N2 - 1))
                    sum_x_unlabeled = x.sum(axis=0).reshape((1, -1))
                    sum_prod_x_unlabeled = mdp.utils.mult(x.T, x)
                    node_weights_complete_2 = w22 * (N2 - 1) * C
                    self._covdcovmtx.update_clustered_homogeneous_block_sizes(
                        x, weight=node_weights_complete_2, block_size=N2)
                    if verbose:
                        print("w22=", w22, "node_weights_complete_2*N2=",
                              node_weights_complete_2 * N2)
                        print("After adding complete 2: num_samples=",
                              self._covdcovmtx.num_samples)
                        print("num_diffs=", self._covdcovmtx.num_diffs)
                        print(" (Diag(C')/num_diffs.avg)**0.5 =",
                              ((numx.diagonal(self._covdcovmtx.sum_prod_diffs)
                                / self._covdcovmtx.num_diffs).mean()) ** 0.5)
                        print("")

                    # Connections between labeled and unlabeled samples
                    # Accounts for transitions in both directions
                    w12 = (2 * 0.5 * v_norm * (1 / (N1_norm - 1 + v_norm * N2)
                                               + 1 / (N1_norm +
                                                      v_norm * (N2 - 1))))
                    if verbose:
                        print("(twice) w12=", w12)
                    sum_prod_diffs_mixed = \
                        w12 * (N1 * sum_prod_x_unlabeled -
                        (mdp.utils.mult(sum_x_labeled.T, sum_x_unlabeled) +
                         mdp.utils.mult(sum_x_unlabeled.T, sum_x_labeled)) +
                        N2 * sum_prod_x_labeled)
                    self._covdcovmtx.sum_prod_diffs += sum_prod_diffs_mixed
                    # w12 is already counted twice
                    self._covdcovmtx.num_diffs += C * N1_norm * N2 * w12
                    if verbose:
                        print(" (Diag(mixed)/num_diffs.avg)**0.5 =",
                              ((numx.diagonal(sum_prod_diffs_mixed) /
                                  (C * N1_norm * N2 * w12)).mean()) ** 0.5)

                    # Additional adjustment for node weights of unlabeled data
                    missing_weight_unlabeled = v - node_weights_complete_2
                    missing_weight_labeled = 1.0 - node_weights_complete_1
                    if verbose:
                        print("missing_weight_unlabeled=",
                              missing_weight_unlabeled)
                        print("Before two final add_samples: num_samples=",
                              self._covdcovmtx.num_samples)
                        print("num_diffs=", self._covdcovmtx.num_diffs)
                    self._covdcovmtx.add_samples(sum_prod_x_unlabeled,
                                                 sum_x_unlabeled, N2,
                                                 missing_weight_unlabeled)
                    self._covdcovmtx.add_samples(sum_prod_x_labeled,
                                                 sum_x_labeled, N1,
                                                 missing_weight_labeled)
                    if verbose:
                        print("Final transformation: num_samples=",
                              self._covdcovmtx.num_samples)
                        print("num_diffs=", self._covdcovmtx.num_diffs)
                        print("v11=%f+%f, v22=%f+%f" %
                              (weight_adjustment,
                               missing_weight_labeled,
                               node_weights_complete_2,
                               missing_weight_unlabeled))
                        print("w11=%f, w22=%f, w12(two ways)=%f" % (w11, w22,
                                                                    w12))
                        print("(N1/C-1)*w11=%f, N2*w12 (one way)=%f" %
                              ((N1 / C - 1) * w11, N2 * w12 / 2))
                        print("(N2-1)*w22*C=%f, N1*w12 (one way)=%f" %
                              ((N2 - 1) * w22 * C, N1 * w12 / 2))
                        print("(Diag(C')/num_diffs.avg)**0.5 =",
                              ((numx.diagonal(self._covdcovmtx.sum_prod_diffs)
                                / self._covdcovmtx.num_diffs).mean()) ** 0.5)
                elif train_mode == 'ignore_data':
                    if verbose:
                        print("Training graph: ignoring data")
                else:
                    ex = "Unknown training method" + train_mode
                    raise ValueError(ex)
            else:
                ex = "Unknown training method" + str(train_mode)
                raise ValueError(ex)

    def _stop_training(self, debug=False, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print("stop_training: self.block_size=", self.block_size)
            print("self._covdcovmtx.num_samples = ",
                  self._covdcovmtx.num_samples)
            print("self._covdcovmtx.num_diffs= ", self._covdcovmtx.num_diffs)
        cov_mtx, self.avg, dcov_mtx = self._covdcovmtx.fix()

        if verbose:
            print("Finishing GSFA training: ", self._covdcovmtx.num_samples)
            print(" num_samples, and ", self._covdcovmtx.num_diffs,
                  " num_diffs")
            print("DCov[0:3,0:3] is", self.dcov_mtx[0:3, 0:3])

        rng = self._set_range()
        # Solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        try:
            if verbose:
                print("***Range used=", rng)
            self.d, self.sf = self._symeig(dcov_mtx, cov_mtx,
                                           range=rng, overwrite=(not debug))
            d = self.d
            # check that we get only non-negative eigenvalues
            if d.min() < 0:
                raise SymeigException("Got negative eigenvalues: %s." % str(d))
        except SymeigException as exception:
            ex = str(exception) + "\n Covariance matrices may be singular."
            raise NodeException(ex)

        if not debug:
            del self._covdcovmtx
            self._covdcovmtx = None

        self._bias = mult(self.avg, self.sf)
        if verbose:
            print("shape of GSFANode.sf is=", self.sf.shape)

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
        """ This function uses a pseudoinverse of the matrix sf to approximate
         an inverse to the transformation.
        """
        if self.pinv is None:
            self.pinv = pinv(self.sf)
        return mult(y, self.pinv) + self.avg

    def _set_range(self):
        if self.output_dim is not None and (self.output_dim <= self.input_dim
                                            or self.input_dim is None):
            # (eigenvalues sorted in ascending order)
            rng = (1, self.output_dim)
        else:
            # otherwise, keep all output components
            rng = None
            self.output_dim = self.input_dim
        return rng


###############################################################################
#                              HELPER FUNCTIONS                               #
###############################################################################

def graph_delta_values(y, edge_weights):
    """ Computes delta values from an arbitrary graph as in the objective
    function of GSFA. The feature vectors are not normalized to weighted
    unit variance or weighted zero mean.
    """
    R = 0
    deltas = 0
    for (i, j) in edge_weights.keys():
        w_ij = edge_weights[(i, j)]
        deltas += w_ij * (y[j] - y[i]) ** 2
        R += w_ij
    return deltas / R


def comp_delta(x):
    """ Computes delta values as in the objective function of SFA.
    The feature vectors are not normalized to unit variance or zero mean.
    """
    xderiv = x[1:, :] - x[:-1, :]
    return (xderiv ** 2).mean(axis=0)


def Hamming_weight(integer_list):
    """ Computes the Hamming weight of an integer or a list of integers (number
     of bits equal to one).
    """
    if isinstance(integer_list, list):
        weights = [Hamming_weight(k) for k in integer_list]
        return weights
    elif isinstance(integer_list, int):
        w = 0
        n = integer_list
        while n > 0:
            if n % 2:
                w += 1
            n = n // 2
        return w
    else:
        er = "unsupported input type for Hamming_weight:" + str(integer_list)
        raise ValueError(er)


class CovDCovMatrix(object):
    """Special purpose class to compute the covariance/second moment matrices
    used by GSFA. It supports efficiently training methods for various graphs:
    e.g., clustered, serial, mixed.
    Joint computation of these matrices is typically more efficient than
    their separate computation.
    """
    def __init__(self, verbose=False):
        """Variable descriptions:
            sum_x: a vector with the sum of all data samples
            sum_prod_x: a matrix with sum of all samples multiplied by their
                transposes
            num_samples: the total weighted number of samples
            sum_prod_diffs: a matrix with sum of all sample differences
                multiplied by their transposes
            num_diffs: the total weighted number of sample differences
            verbose: a Boolean verbosity parameter

        The following variables are available after fix() has been called.
            cov_mtx: the resulting covariance matrix of the samples
            avg: the average sample
            dcov_mtx: the resulting second-moment matrix of the sample
            differences
        """
        self.sum_x = None
        self.sum_prod_x = None
        self.num_samples = 0
        self.sum_prod_diffs = None
        self.num_diffs = 0
        self.verbose = verbose

        # Variables used to store the final matrices
        self.cov_mtx = None
        self.avg = None
        self.dcov_mtx = None

    def add_samples(self, sum_prod_x, sum_x, num_samples, weight=1.0):
        """ The given sample information (sum_prod_x, sum_x, num_samples) is
        added to the cumulative
        computation of the covariance matrix.
        """
        weighted_sum_x = sum_x * weight
        weighted_sum_prod_x = sum_prod_x * weight
        weighted_num_samples = num_samples * weight

        if self.sum_prod_x is None:
            self.sum_prod_x = weighted_sum_prod_x
            self.sum_x = weighted_sum_x
        else:
            self.sum_prod_x = self.sum_prod_x + weighted_sum_prod_x
            self.sum_x = self.sum_x + weighted_sum_x

        self.num_samples = self.num_samples + weighted_num_samples

    def add_diffs(self, sum_prod_diffs, num_diffs, weight=1.0):
        """ The given sample differences information (sum_prod_diffs,
        num_diffs) is added to the cumulative computation of the second-moment
        differences matrix.
        """
        weighted_sum_prod_diffs = sum_prod_diffs * weight
        weighted_num_diffs = num_diffs * weight

        if self.sum_prod_diffs is None:
            self.sum_prod_diffs = weighted_sum_prod_diffs
        else:
            self.sum_prod_diffs = self.sum_prod_diffs + weighted_sum_prod_diffs

        self.num_diffs = self.num_diffs + weighted_num_diffs

    def update_unlabeled(self, x, weight=1.0):
        """ Add unlabeled samples to the covariance matrix
        (DCov remains unmodified).
        """
        num_samples = x.shape[0]

        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

    def update_regular(self, x, weight=1.0):
        """This is equivalent to regular SFA training (except for the final
        feature scale).
        """
        num_samples = x.shape[0]

        # Update Cov Matrix
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix
        diffs = x[1:, :] - x[:-1, :]
        num_diffs = num_samples - 1
        sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
        self.add_diffs(sum_prod_diffs, num_diffs, weight)

    def update_graph(self, x, node_weights=None, edge_weights=None,
                     weight=1.0):
        """Updates the covariance/second moment matrices using an user-provided
         graph specified by (x, node weights, edge weights, and a global
         weight).

         Usually sum(node_weights) = num_samples.
         """
        num_samples, dim = x.shape

        if node_weights is None:
            node_weights = numx.ones(num_samples)

        if len(node_weights) != num_samples:
            er = "Node weights should have the same length " + \
                 "%d as the number of samples %d" % (len(node_weights),
                                                     num_samples)
            raise TrainingException(er)

        if edge_weights is None:
            er = "edge_weights should be a dictionary with entries: " + \
                 "d[(i,j)] = w_{i,j} or an NxN array"
            raise TrainingException(er)

        if isinstance(edge_weights, numx.ndarray):
            # TODO: eventually make sure edge_weights are symmetric
            # TODO: eventually make sure consistency restriction is fulfilled
            if edge_weights.shape != (num_samples, num_samples):
                er = "Shape of edge_weights should be " + \
                     "(%d, %d)" % (num_samples, num_samples) + \
                     " but is (%d, %d)" % (edge_weights.shape[0],
                                           edge_weights.shape[1])
                raise TrainingException(er)

        node_weights_column = node_weights.reshape((num_samples, 1))
        # Update Cov Matrix
        weighted_x = x * node_weights_column

        weighted_sum_x = weighted_x.sum(axis=0)
        weighted_sum_prod_x = mdp.utils.mult(x.T, weighted_x)
        weighted_num_samples = node_weights.sum()
        self.add_samples(weighted_sum_prod_x, weighted_sum_x,
                         weighted_num_samples, weight=weight)

        # Update DCov Matrix
        if isinstance(edge_weights, numx.ndarray):
            weighted_num_diffs = edge_weights.sum()  # normalization constant R
            prod1 = weighted_sum_prod_x  # TODO: eventually check these
            # equations, they might only work if Q==R
            prod2 = mdp.utils.mult(mdp.utils.mult(x.T, edge_weights), x)
            weighted_sum_prod_diffs = 2 * prod1 - 2 * prod2
            self.add_diffs(weighted_sum_prod_diffs, weighted_num_diffs,
                           weight=weight)
        else:
            num_diffs = len(edge_weights)
            diffs = numx.zeros((num_diffs, dim))
            weighted_diffs = numx.zeros((num_diffs, dim))
            weighted_num_diffs = 0
            for ii, (i, j) in enumerate(edge_weights.keys()):
                diff = x[j, :] - x[i, :]
                diffs[ii] = diff
                w_ij = edge_weights[(i, j)]
                weighted_diff = diff * w_ij
                weighted_diffs[ii] = weighted_diff
                weighted_num_diffs += w_ij

            weighted_sum_prod_diffs = mdp.utils.mult(diffs.T, weighted_diffs)
            self.add_diffs(weighted_sum_prod_diffs, weighted_num_diffs,
                           weight=weight)

    def update_graph_old(self, x, node_weights=None, edge_weights=None,
                         weight=1.0):
        """This method performs the same task as update_graph. It is slower
        than update_graph because it has not been optimized. Thus, it is
        mainly useful to verify the correctness of update_graph.
        """
        num_samples, dim = x.shape

        if node_weights is None:
            node_weights = numx.ones(num_samples)

        if len(node_weights) != num_samples:
            er = "Node weights should be the same length " + \
                 "%d as the number of samples %d" % (len(node_weights),
                                                     num_samples)
            raise TrainingException(er)

        if edge_weights is None:
            er = "edge_weights should be a dictionary with entries: " + \
                 "d[(i,j)] = w_{i,j} or an NxN array"
            raise TrainingException(er)

        if isinstance(edge_weights, numx.ndarray):
            if edge_weights.shape == (num_samples, num_samples):
                e_w = {}
                for i in range(num_samples):
                    for j in range(num_samples):
                        if edge_weights[i, j] != 0:
                            e_w[(i, j)] = edge_weights[i, j]
                edge_weights = e_w
            else:
                er = "edge_weights.shape should be (%d,%d) but is (%d,%d)" % \
                     (num_samples, num_samples, edge_weights.shape[0],
                      edge_weights.shape[1])
                raise TrainingException(er)
        node_weights_column = node_weights.reshape((num_samples, 1))
        # Update Cov Matrix
        weighted_x = x * node_weights_column

        weighted_sum_x = weighted_x.sum(axis=0)
        weighted_sum_prod_x = mdp.utils.mult(x.T, weighted_x)
        weighted_num_samples = node_weights.sum()
        self.add_samples(weighted_sum_prod_x, weighted_sum_x,
                         weighted_num_samples, weight=weight)

        # Update DCov Matrix
        num_diffs = len(edge_weights)
        diffs = numx.zeros((num_diffs, dim))
        weighted_diffs = numx.zeros((num_diffs, dim))
        weighted_num_diffs = 0
        for ii, (i, j) in enumerate(edge_weights.keys()):
            diff = x[j, :] - x[i, :]
            diffs[ii] = diff
            w_ij = edge_weights[(i, j)]
            weighted_diff = diff * w_ij
            weighted_diffs[ii] = weighted_diff
            weighted_num_diffs += w_ij

        weighted_sum_prod_diffs = mdp.utils.mult(diffs.T, weighted_diffs)
        self.add_diffs(weighted_sum_prod_diffs, weighted_num_diffs,
                       weight=weight)

    def update_mirroring_sliding_window(self, x, weight=1.0,
                                        window_halfwidth=2):
        """ Note: this method makes sense according to the consistency
        restriction for "larger" windows.
        """
        num_samples, dim = x.shape
        # window_halfwidth is too long to write it complete each time
        width = window_halfwidth
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples" % \
                 (width, num_samples)
            raise TrainingException(ex)

        # Update Cov Matrix. All samples have same weight
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix. First mirror the borders
        x_mirror = numx.zeros((num_samples + 2 * width, dim))
        x_mirror[width:-width] = x  # center part
        x_mirror[0:width, :] = x[0:width, :][::-1, :]  # first end
        x_mirror[-width:, :] = x[-width:, :][::-1, :]  # second end

        # Center part
        x_full = x
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)

        Aacc123 = numx.zeros((dim, dim))
        for i in range(0, 2 * width):
            Aacc123 += (i + 1) * mdp.utils.mult(x_mirror[i:i + 1, :].T,
                                                x_mirror[i:i + 1, :])

        for i in range(num_samples, num_samples + 2 * width):
            Aacc123 += (num_samples + 2 * width - i) * mdp.utils.mult(
                x_mirror[i:i + 1, :].T, x_mirror[i:i + 1, :])
        # intermediate values of x, which are connected 2*width+1 times
        x_middle = x_mirror[2 * width:-2 * width, :]
        Aacc123 += (2 * width + 1) * mdp.utils.mult(x_middle.T, x_middle)

        b = numx.zeros((num_samples + 1 + 2 * width, dim))
        b[1:] = x_mirror.cumsum(axis=0)
        B = b[2 * width + 1:] - b[0:-2 * width - 1]
        Bprod = mdp.utils.mult(x_full.T, B)

        sum_prod_diffs_full = (2 * width + 1) * sum_prod_x_full + Aacc123 - \
                              Bprod - Bprod.T
        num_diffs = num_samples * (2 * width)  # removed zero differences
        self.add_diffs(sum_prod_diffs_full, num_diffs, weight)

    def update_slow_mirroring_sliding_window(self, x, weight=1.0,
                                             window_halfwidth=2):
        """ This is an unoptimized version of update_mirroring_sliding_window.
        """
        num_samples, dim = x.shape
        width = window_halfwidth
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d " % width + \
                 "not supported for %d samples" % num_samples
            raise TrainingException(ex)

        # Update Cov Matrix. All samples have same weight
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix. window = numx.ones(2*width+1), rectangular window
        x_mirror = numx.zeros((num_samples + 2 * width, dim))
        x_mirror[width:-width] = x  # center part
        x_mirror[0:width, :] = x[0:width, :][::-1, :]  # start of the sequence
        x_mirror[-width:, :] = x[-width:, :][::-1, :]  # end of the sequence

        for offset in range(-width, width + 1):
            if offset == 0:
                pass
            else:
                diffs = x_mirror[offset + width:offset + width +
                                                num_samples, :] - x

                sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                num_diffs = len(diffs)
                self.add_diffs(sum_prod_diffs, num_diffs, weight)

    def update_slow_truncating_sliding_window(self, x, weight=1.0,
                                              window_halfwidth=2):
        """ Truncating Window (original slow/reference version). """
        num_samples, dim = x.shape
        width = window_halfwidth
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d " % width + \
                 "not supported for %d samples" % num_samples
            raise ValueError(ex)

        # Update Cov Matrix. All samples have same weight
        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # Update DCov Matrix. window = numx.ones(2*width+1), rectangular window
        x_extended = numx.zeros((num_samples + 2 * width, dim))
        # Center part is preserved, extreme samples are zero
        x_extended[width:-width] = x

        # Negative offset is not considered because it is equivalent to the
        # positive one, thereore the factor 2
        for offset in range(1, width + 1):
            diffs = x_extended[offset + width:width + num_samples, :] - \
                    x[0:-offset, :]
            sum_prod_diffs = 2 * mdp.utils.mult(diffs.T, diffs)
            num_diffs = 2 * (num_samples - offset)
            self.add_diffs(sum_prod_diffs, num_diffs, weight)

    def update_fast_sliding_window(self, x, weight=1.0, window_halfwidth=2):
        """ Sliding window with node-weight correction. """
        num_samples, dim = x.shape
        width = window_halfwidth
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples" % (width,
                                                                        num_samples)
            raise ValueError(ex)

        # MOST CORRECT VERSION
        x_sel = x + 0.0
        w_up = numx.arange(width, 2 * width) / (2.0 * width)
        w_up = w_up.reshape((width, 1))
        w_down = numx.arange(2 * width - 1, width - 1, -1) / (2.0 * width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] = x_sel[0:width, :] * w_up
        x_sel[-width:, :] = x_sel[-width:, :] * w_down

        sum_x = x_sel.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x_sel.T, x)
        self.add_samples(sum_prod_x, sum_x,
                         num_samples - (0.5 * window_halfwidth - 0.5), weight)

        # Update DCov Matrix. First we compute the borders
        # Left border
        for i in range(0, width):  # [0, width -1]
            diffs = x[0:width + i + 1, :] - x[i, :]
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1  # removed zero differences
            # print "N1=", num_diffs
            # print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.add_diffs(sum_prod_diffs, num_diffs, weight)
        # Right border
        for i in range(num_samples - width, num_samples):
            diffs = x[i - width:num_samples, :] - x[i, :]
            sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
            num_diffs = len(diffs) - 1  # removed zero differences
            # print "N2=", num_diffs
            # print "sum_prod_diffs[0]=", sum_prod_diffs[0]
            self.add_diffs(sum_prod_diffs, num_diffs, weight)

        # Center part
        x_full = x[width:num_samples - width, :]
        sum_prod_x_full = mdp.utils.mult(x_full.T, x_full)

        Aacc123 = numx.zeros((dim, dim))
        for i in range(0, 2 * width):  # [0, 2*width-1]
            Aacc123 += (i + 1) * mdp.utils.mult(x[i:i + 1, :].T, x[i:i + 1, :])

        for i in range(num_samples - 2 * width, num_samples):
            Aacc123 += (num_samples - i) * mdp.utils.mult(x[i:i + 1, :].T,
                                                          x[i:i + 1, :])

        # intermediate values of x, which are connected 2*width+1 times
        x_middle = x[2 * width:num_samples - 2 * width, :]

        Aacc123 += (2 * width + 1) * mdp.utils.mult(x_middle.T, x_middle)

        b = numx.zeros((num_samples + 1, dim))
        b[1:] = x.cumsum(axis=0)
        B = b[2 * width + 1:] - b[0:-2 * width - 1]
        Bprod = mdp.utils.mult(x_full.T, B)
        sum_prod_diffs_full = (2 * width + 1) * sum_prod_x_full + Aacc123 - \
                              Bprod - Bprod.T
        num_diffs = (num_samples - 2 * width) * (2 * width)
        self.add_diffs(sum_prod_diffs_full, num_diffs, weight)

    def update_sliding_window(self, x, weight=1.0, window_halfwidth=2):
        num_samples = x.shape[0]
        width = window_halfwidth
        if 2 * width >= num_samples:
            ex = "window_halfwidth %d not supported for %d samples" % \
                 (width, num_samples)
            raise ValueError(ex)

        # MOST CORRECT VERSION
        x_sel = x + 0.0
        w_up = numx.arange(width, 2 * width) / (2.0 * width)
        w_up = w_up.reshape((width, 1))
        w_down = numx.arange(2 * width - 1, width - 1, -1) / (2.0 * width)
        w_down = w_down.reshape((width, 1))
        x_sel[0:width, :] = x_sel[0:width, :] * w_up
        x_sel[-width:, :] = x_sel[-width:, :] * w_down

        sum_x = x_sel.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x_sel.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples -
                         (0.5 * window_halfwidth - 0.5), weight)

        # This can be made faster (twice) due to symmetry
        for offset in range(-width, width + 1):
            if offset == 0:
                pass
            else:
                if offset > 0:
                    diffs = x[offset:, :] - x[0:num_samples - offset, :]
                    sum_prod_diffs = mdp.utils.mult(diffs.T, diffs)
                    num_diffs = len(diffs)
                    self.add_diffs(sum_prod_diffs, num_diffs, weight)

    # Add samples belonging to a serial training graph
    def update_serial(self, x, block_size, weight=1.0):
        num_samples, dim = x.shape
        if block_size is None:
            er = "block_size must be specified"
            raise TrainingException(er)

        if isinstance(block_size, numx.ndarray):
            err = "Inhomogeneous block sizes not yet supported"
            raise ValueError(err)
        elif isinstance(block_size, list):
            block_size_0 = block_size[0]
            for bs in block_size:
                if bs != block_size_0:
                    er = "for serial graph all groups must have same group" + \
                         "size (block_size constant), but " + \
                         str(bs) + "!=" + str(block_size_0)
                    raise ValueError(er)
            block_size = block_size_0

        if num_samples % block_size > 0:
            err = "num_samples is not a multiple of block_size"
            raise ValueError(err)
        num_blocks = num_samples // block_size

        # Correlation Matrix. Computing sum of outer products (the easy part)
        xp = x[block_size:num_samples - block_size]
        x_b_ini = x[0:block_size]
        x_b_end = x[num_samples - block_size:]
        sum_x = x_b_ini.sum(axis=0) + 2 * xp.sum(axis=0) + x_b_end.sum(axis=0)

        sum_prod_x = mdp.utils.mult(x_b_ini.T, x_b_ini) + \
                     2 * mdp.utils.mult(xp.T, xp) + \
                     mdp.utils.mult(x_b_end.T, x_b_end)
        num_samples = 2 * block_size + 2 * (num_samples - 2 * block_size)

        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # DCorrelation Matrix. Compute medias signal
        media = numx.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i * block_size:(i + 1) * block_size].sum(axis=0) * \
                       (1.0 / block_size)

        media_a = media[0:-1]
        media_b = media[1:]
        sum_prod_mixed_meds = (mdp.utils.mult(media_a.T, media_b) +
                               mdp.utils.mult(media_b.T, media_a))

        num_diffs = block_size * (num_blocks - 1)

        sum_prod_diffs = (block_size * sum_prod_x - block_size * block_size
                          * sum_prod_mixed_meds) * (1.0 / block_size)
        # The factor 2 accounts for both directions
        self.add_diffs(2 * sum_prod_diffs, 2 * num_diffs, weight)

    # Weight should refer to node weights
    def update_clustered(self, x, block_sizes=None, weight=1.0,
                         include_self_loops=True):
        num_samples = x.shape[0]

        if isinstance(block_sizes, int):
            self.update_clustered_homogeneous_block_sizes(
                x, weight=weight, block_size=block_sizes,
                include_self_loops=include_self_loops)
            return

        if block_sizes is None:
            er = "block_size is not specified"
            raise TrainingException(er)

        if num_samples != numx.array(block_sizes).sum():
            err = "Inconsistency error: num_samples (%d)" % num_samples + \
                "is not equal to sum of block_sizes: %d" % block_sizes
            raise ValueError(err)

        counter_sample = 0
        for block_size in block_sizes:
            normalized_weight = weight
            self.update_clustered_homogeneous_block_sizes(
                x[counter_sample:counter_sample + block_size, :],
                weight=normalized_weight, block_size=block_size,
                include_self_loops=include_self_loops)
            counter_sample += block_size

    def update_clustered_homogeneous_block_sizes(self, x, weight=1.0,
                                                 block_size=None,
                                                 include_self_loops=True):
        if self.verbose:
            print("update_clustered_homogeneous_block_sizes ")
        if block_size is None:
            er = "block_size is not specified"
            raise TrainingException(er)

        if isinstance(block_size, numx.ndarray):
            er = "inhomogeneous block sizes are not supported by this function"
            raise TrainingException(er)

        # Assuming block_size is an integer:
        num_samples, dim = x.shape
        if num_samples % block_size > 0:
            err = "num_samples (%d) is not a multiple of block_size (%d)" % \
                  (num_samples, block_size)
            raise ValueError(err)
        num_blocks = num_samples // block_size

        sum_x = x.sum(axis=0)
        sum_prod_x = mdp.utils.mult(x.T, x)
        self.add_samples(sum_prod_x, sum_x, num_samples, weight)

        # DCorrelation Matrix. Compute medias signal
        media = numx.zeros((num_blocks, dim))
        for i in range(num_blocks):
            media[i] = x[i * block_size:(i + 1) * block_size].sum(axis=0) * \
                       (1.0 / block_size)

        sum_prod_meds = mdp.utils.mult(media.T, media)
        num_diffs = num_blocks * block_size
        if self.verbose:
            print("num_diffs in block:", num_diffs,
                  " num_samples:", num_samples)
        if include_self_loops:
            sum_prod_diffs = 2.0 * block_size * (sum_prod_x - block_size *
                                                 sum_prod_meds) / block_size
        else:
            sum_prod_diffs = 2.0 * block_size * (sum_prod_x - block_size *
                                            sum_prod_meds) / (block_size - 1)

        self.add_diffs(sum_prod_diffs, num_diffs, weight)
        if self.verbose:
            print("(Diag(complete)/num_diffs.avg)**0.5 =",
                  ((numx.diagonal(sum_prod_diffs) / num_diffs).mean()) ** 0.5)

    def update_compact_classes(self, x, block_sizes=None, Jdes=None):
        num_samples = x.shape[0]

        if self.verbose:
            print("block_sizes=", block_sizes, type(block_sizes))
        if isinstance(block_sizes, list):
            block_sizes = numx.array(block_sizes)

        if isinstance(block_sizes, numx.ndarray):
            if len(block_sizes) > 1:
                if block_sizes.var() > 0:
                    er = "for compact_classes all groups must have the " + \
                         "same number of elements (block_sizes)"
                    raise ValueError(er)
                else:
                    block_size = block_sizes[0]
            else:
                block_size = block_sizes[0]
        elif block_sizes is None:
            er = "block_size not specified"
            raise TrainingException(er)
        else:
            block_size = block_sizes

        if num_samples % block_size != 0:
            err = "num_samples (%d) should be a multiple of block_size: %d" % \
                  (num_samples, block_sizes)
            raise ValueError(err)

        num_classes = num_samples // block_size
        J = int(numx.log2(num_classes))
        if Jdes is None:
            Jdes = J
        extra_label = Jdes - J

        if self.verbose:
            print("Besides J=%d labels, also adding %d labels" % (J,
                                                                  extra_label))

        if num_classes != 2 ** J:
            err = "num_clases %d is probably not a power of 2" % num_classes
            raise ValueError(err)

        N = num_samples
        labels = numx.zeros((N, J + extra_label))
        for j in range(J):
            labels[:, j] = (numx.arange(N) // block_size //
                            (2 ** (J - j - 1)) % 2) * 2 - 1
        eigenvalues = numx.concatenate(([1.0] * (J - 1),
                                        numx.arange(1.0, 0.0,
                                                    -1.0 / (extra_label + 1))))

        n_taken = [2 ** k for k in range(J)]
        n_free = list(set(range(num_classes)) - set(n_taken))
        n_free_weights = Hamming_weight(n_free)
        order = numx.argsort(n_free_weights)[::-1]

        for j in range(extra_label):
            digit = n_free[order[j]]
            label = numx.ones(N)
            for c in range(J):
                if (digit >> c) % 2:
                    label *= labels[:, c]
            if n_free_weights[order[j]] % 2 == 0:
                label *= -1
            labels[:, J + j] = label

        eigenvalues = numx.array(eigenvalues)

        eigenvalues /= eigenvalues.sum()
        if self.verbose:
            print("Eigenvalues:", eigenvalues)
            print("Eigenvalues normalized:", eigenvalues)
            for j in range(J + extra_label):
                print("labels[%d]=" % j, labels[:, j])

        for j in range(J + extra_label):
            set10 = x[labels[:, j] == -1]
            # first cluster
            self.update_clustered_homogeneous_block_sizes(
                set10, weight=eigenvalues[j], block_size=N // 2)
            set10 = x[labels[:, j] == 1]
            # second cluster
            self.update_clustered_homogeneous_block_sizes(
                set10, weight=eigenvalues[j], block_size=N // 2)

    def add_cov_dcov_matrix(self, cov_dcov_mat, adding_weight=1.0,
                            own_weight=1.0):
        if self.sum_prod_x is None:
            self.sum_prod_x = cov_dcov_mat.sum_prod_x * adding_weight
            self.sum_x = cov_dcov_mat.sum_x * adding_weight
        else:
            self.sum_prod_x = self.sum_prod_x * own_weight + \
                              cov_dcov_mat.sum_prod_x * adding_weight
            self.sum_x = self.sum_x * own_weight + \
                         cov_dcov_mat.sum_x * adding_weight
        self.num_samples = self.num_samples * own_weight + \
                           cov_dcov_mat.num_samples * adding_weight
        if self.sum_prod_diffs is None:
            self.sum_prod_diffs = cov_dcov_mat.sum_prod_diffs * adding_weight
        else:
            self.sum_prod_diffs = self.sum_prod_diffs * own_weight + \
                                  cov_dcov_mat.sum_prod_diffs * adding_weight
        self.num_diffs = self.num_diffs * own_weight + \
                         cov_dcov_mat.num_diffs * adding_weight

    def fix(self, divide_by_num_samples_or_differences=True):
        if self.verbose:
            print("Fixing CovDCovMatrix")

        avg_x = self.sum_x * (1.0 / self.num_samples)

        prod_avg_x = numx.outer(avg_x, avg_x)
        if divide_by_num_samples_or_differences:
            # as specified by the theory on training graphs
            cov_x = (self.sum_prod_x -
                     self.num_samples * prod_avg_x) / (1.0 * self.num_samples)
        else:  # standard unbiased estimation used by standard SFA
            cov_x = (self.sum_prod_x -
                     self.num_samples * prod_avg_x) / (self.num_samples - 1.0)

        # Finalize covariance matrix of dx
        if divide_by_num_samples_or_differences or True:
            cov_dx = self.sum_prod_diffs / (1.0 * self.num_diffs)
        else:
            cov_dx = self.sum_prod_diffs / (self.num_diffs - 1.0)

        self.cov_mtx = cov_x
        self.avg = avg_x
        self.dcov_mtx = cov_dx

        if self.verbose:
            print("Finishing training CovDcovMtx:", self.num_samples,
                  "num_samples, and", self.num_diffs, "num_diffs")
            print("Avg[0:3] is", self.avg[0:4])
            print("Prod_avg_x[0:3,0:3] is", prod_avg_x[0:3, 0:3])
            print("Cov[0:3,0:3] is", self.cov_mtx[0:3, 0:3])
            print("DCov[0:3,0:3] is", self.dcov_mtx[0:3, 0:3])
            # print("AvgDiff[0:4] is", avg_diff[0:4])
            # print("Prod_avg_diff[0:3,0:3] is", prod_avg_diff[0:3,0:3])
            print("Sum_prod_diffs[0:3,0:3] is", self.sum_prod_diffs[0:3, 0:3])
            # print("exp_prod_diffs[0:3,0:3] is", exp_prod_diffs[0:3,0:3])
        return self.cov_mtx, self.avg, self.dcov_mtx


class iGSFANode(mdp.Node):
    """This node implements "information-preserving graph-based SFA (iGSFA)",
    which is the main component of hierarchical iGSFA (HiGSFA).

    For further information, see: Escalante-B., A.-N. and Wiskott, L.,
    "Improved graph-based {SFA}: Information preservation complements the
    slowness principle", e-print arXiv:1601.03945,
    http://arxiv.org/abs/1601.03945, 2017.
    """
    def __init__(self, pre_expansion_node_class=None,
                 pre_expansion_out_dim=None, expansion_funcs=None,
                 expansion_output_dim=None, expansion_starting_point=None,
                 max_length_slow_part=None, slow_feature_scaling_method=None,
                 delta_threshold=1.999, reconstruct_with_sfa=False,
                 verbose=False, input_dim=None, output_dim=None,
                 dtype=None, **argv):
        """ Initializes the iGSFA node.
        pre_expansion_node_class: a node class. An instance of this class is
            used to filter the data before the expansion.
        pre_expansion_out_dim: the output dimensionality of the
            above-mentioned node.
        expansion_funcs: a list of expansion functions to be applied before
            GSFA.
        expansion_output_dim: this parameter is used to specify an output
            dimensionality for some expansion functions.
        expansion_starting_point: this parameter is also used by some specific
            expansion functions.
        max_length_slow_part: fixes an upper bound to the size of the slow
            part, which is convenient for computational reasons.
        slow_feature_scaling_method: the method used to scale the slow
            features. Valid entries are: None, "sensitivity_based" (default),
            "data_dependent", and "QR_decomposition".
        delta_threshold: this parameter has two different meanings depending
            on its type. If it is real valued (e.g., 1.99), it determines the
            parameter Delta_threshold, which is used to decide how many slow
            features are preserved, depending on their delta values. If it is
            integer (e.g., 20), it directly specifies the exact size of the
            slow part.
        reconstruct_with_sfa: this Boolean parameter indicates whether the
            slow part is removed from the input before PCA is applied.

        More information about parameters 'expansion_funcs' and
            'expansion_starting_point' can be found in the documentation of
            GeneralExpansionNode.

        Note: Training sometimes finishes after a single call to the train
            method, unless multi-train is enabled (default). Multi-train is
            enabled by setting reconstruct_with_sfa=False and
            slow_feature_scaling_method in [None, "data_dependent"].
            This is necessary to support weight sharing in iGSFA layers
            (convolutional iGSFA layers).
        """
        super(iGSFANode, self).__init__(input_dim=input_dim,
                                        output_dim=output_dim,
                                        dtype=dtype, **argv)
        # Type of node used to expand the data
        self.pre_expansion_node_class = pre_expansion_node_class
        self.pre_expansion_node = None  # Node that expands the input data
        self.pre_expansion_output_dim = pre_expansion_out_dim
        # Expanded dimensionality
        self.expansion_output_dim = expansion_output_dim
        # Initial parameters for the expansion function
        self.expansion_starting_point = expansion_starting_point

        # creates an expansion node
        if expansion_funcs and self.expansion_output_dim is not None:
            self.exp_node = GeneralExpansionNode(
                funcs=expansion_funcs, output_dim=self.expansion_output_dim,
                starting_point=self.expansion_starting_point)
        elif expansion_funcs:
            self.exp_node = GeneralExpansionNode(
                funcs=expansion_funcs,
                starting_point=self.expansion_starting_point)
        else:
            self.exp_node = None

        self.sfa_node = None
        self.pca_node = None
        self.lr_node = None
        # hard upper limit to the size of the slow part
        self.max_length_slow_part = max_length_slow_part

        # Parameter that defines the size of the slow part. Its meaning
        # depends on wheather it is an integer or a float
        self.delta_threshold = delta_threshold
        # Indicates whether (nonlinear) SFA components are used for
        # reconstruction
        self.reconstruct_with_sfa = reconstruct_with_sfa
        # Indicates how to scale the slow part
        self.slow_feature_scaling_method = slow_feature_scaling_method

        # Default value when none is explicity provided to the class methods
        self.verbose = verbose

        # Dimensionality of the data after the expansion function
        self.expanded_dim = None

        # The following variables are for internal use only (available after
        # training on a single batch only)
        self.magn_n_sfa_x = None
        self.num_sfa_features_preserved = None
        self.x_mean = None
        self.sfa_x_mean = None
        self.sfa_x_std = None
        self.evar = None
        # The following variables are for internal use only, by the QR slow
        # feature scaling method
        self.Q = None
        self.R = None
        self.Rpinv = None

    @staticmethod
    def is_trainable():
        return True

    # TODO: should train_mode be renamed training_mode?
    def _train(self, x, block_size=None, train_mode=None, node_weights=None,
               edge_weights=None, verbose=None, **argv):
        """Trains an iGSFA node on data 'x'

        The parameters:  block_size, train_mode, node_weights, and edge_weights
        are passed to the training function of the corresponding gsfa node
        inside iGSFA (node.gsfa_node).
        """
        self.input_dim = x.shape[1]
        if verbose is None:
            verbose = self.verbose

        if self.output_dim is None:
            self.output_dim = self.input_dim

        if verbose:
            print("Training iGSFANode...")

        if (not self.reconstruct_with_sfa) and \
                (self.slow_feature_scaling_method in [None, "data_dependent"]):
            self.multiple_train(x, block_size=block_size,
                                train_mode=train_mode,
                                node_weights=node_weights,
                                edge_weights=edge_weights)
            return

        if (not self.reconstruct_with_sfa) and \
                (self.slow_feature_scaling_method not in [None,
                                                          "data_dependent"]):
            er = "'reconstruct_with_sfa' " + str(self.reconstruct_with_sfa) + \
                 ") should be True when the scaling method (" + \
                 str(self.slow_feature_scaling_method) + \
                 ") is neither 'None' not 'data_dependent'"
            raise TrainingException(er)
        # else continue using the regular method:

        # Remove mean before expansion
        self.x_mean = x.mean(axis=0)
        x_zm = x - self.x_mean

        # Reorder or pre-process the data before it is expanded,
        # but only if there is really an expansion
        if self.pre_expansion_node_class and self.exp_node:
            self.pre_expansion_node = self.pre_expansion_node_class(
                output_dim=self.pre_expansion_output_dim)
            # reasonable options for pre_expansion_node_class are GSFANode
            # or WhitheningNode
            self.pre_expansion_node.train(x_zm, block_size=block_size,
                                          train_mode=train_mode)
            self.pre_expansion_node.stop_training()
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        # Expand data
        if self.exp_node:
            if verbose:
                print("expanding x...")
            exp_x = self.exp_node.execute(x_pre_exp)
        else:
            exp_x = x_pre_exp

        self.expanded_dim = exp_x.shape[1]

        if self.max_length_slow_part is None:
            sfa_output_dim = min(self.expanded_dim, self.output_dim)
        else:
            sfa_output_dim = min(self.max_length_slow_part, self.expanded_dim,
                                 self.output_dim)

        if isinstance(self.delta_threshold, int):
            sfa_output_dim = min(sfa_output_dim, self.delta_threshold)
            sfa_output_dim = max(1, sfa_output_dim)

        # Apply SFA to expanded data
        self.sfa_node = GSFANode(output_dim=sfa_output_dim, dtype=self.dtype,
                                 verbose=verbose)
        self.sfa_node.train(exp_x, block_size=block_size,
                            train_mode=train_mode, node_weights=node_weights,
                            edge_weights=edge_weights)  # sfa_node.train_params
        self.sfa_node.stop_training()
        if verbose:
            print("self.sfa_node.d", self.sfa_node.d)

        # Decide how many slow features are preserved (either use
        # Delta_T=delta_threshold when delta_threshold is a float, or
        # preserve delta_threshold features when delta_threshold is an integer)
        if isinstance(self.delta_threshold, float):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = (self.sfa_node.d <=
                                               self.delta_threshold).sum()
        elif isinstance(self.delta_threshold, int):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = self.delta_threshold
            if self.delta_threshold > self.output_dim:
                er = "The provided integer delta_threshold " + \
                     "%d is larger than the output dimensionality %d" % \
                     (self.delta_threshold, self.output_dim)
                raise ValueError(er)
            if self.max_length_slow_part is not None and \
                            self.delta_threshold > self.max_length_slow_part:
                er = "The provided integer delta_threshold " + \
                     "%d" % self.delta_threshold + \
                     " is larger than the given upper bound on the size " + \
                     "of the slow part (max_length_slow_part) %d" % \
                     self.max_length_slow_part
                raise ValueError(er)

        else:
            ex = "Cannot handle type of self.delta_threshold"
            raise ValueError(ex)

        if self.num_sfa_features_preserved > self.output_dim:
            self.num_sfa_features_preserved = self.output_dim

        SFANode_reduce_output_dim(self.sfa_node,
                                  self.num_sfa_features_preserved)
        if verbose:
            print("sfa execute...")
        sfa_x = self.sfa_node.execute(exp_x)

        # normalize sfa_x
        self.sfa_x_mean = sfa_x.mean(axis=0)
        self.sfa_x_std = sfa_x.std(axis=0)
        if verbose:
            print("self.sfa_x_mean=", self.sfa_x_mean)
            print("self.sfa_x_std=", self.sfa_x_std)
        if (self.sfa_x_std == 0).any():
            er = "zero-component detected"
            raise TrainingException(er)
        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std

        if self.reconstruct_with_sfa:
            x_pca = x_zm

            # Approximate input linearly
            if verbose:
                print("training linear regression...")
            self.lr_node = mdp.nodes.LinearRegressionNode()
            # Notice that the input data is "x"=n_sfa_x and the output
            # to learn is "y" = x_pca
            self.lr_node.train(n_sfa_x, x_pca)
            self.lr_node.stop_training()
            x_pca_app = self.lr_node.execute(n_sfa_x)
            x_app = x_pca_app
        else:
            x_app = numx.zeros_like(x_zm)

        # Remove linear approximation
        sfa_removed_x = x_zm - x_app

        # TODO:Compute variance removed by linear approximation
        if verbose:
            print("ranking method...")
        # A method for feature scaling( +rotation)
        if self.reconstruct_with_sfa and \
                        self.slow_feature_scaling_method == "QR_decomposition":
            # A bias term is included by default, we do not need it
            M = self.lr_node.beta[1:, :].T
            Q, R = numx.linalg.qr(M)
            self.Q = Q
            self.R = R
            self.Rpinv = pinv(R)
            s_n_sfa_x = numx.dot(n_sfa_x, R.T)
        # Another method for feature scaling (no rotation)
        elif self.reconstruct_with_sfa and \
                (self.slow_feature_scaling_method == "sensitivity_based"):
            beta = self.lr_node.beta[1:, :]
            sens = (beta ** 2).sum(axis=1)
            self.magn_n_sfa_x = sens ** 0.5
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
            if verbose:
                print("method: sensitivity_based enforced")
        elif self.slow_feature_scaling_method is None:
            self.magn_n_sfa_x = 1.0
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
            if verbose:
                print("method: constant amplitude for all slow features")
        elif self.slow_feature_scaling_method == "data_dependent":
            if verbose:
                print("skiped data_dependent")
        else:
            er = "unknown slow feature scaling method= " + \
                 str(self.slow_feature_scaling_method) + \
                 " for reconstruct_with_sfa= " + str(self.reconstruct_with_sfa)
            raise ValueError(er)

        print("training PCA...")
        pca_output_dim = self.output_dim - self.num_sfa_features_preserved
        # This allows training of PCA when pca_out_dim is zero
        self.pca_node = mdp.nodes.PCANode(output_dim=max(1, pca_output_dim),
                                          dtype=self.dtype)
        self.pca_node.train(sfa_removed_x)
        self.pca_node.stop_training()
        PCANode_reduce_output_dim(self.pca_node, pca_output_dim,
                                  verbose=False)

        if verbose:
            print("executing PCA...")

        pca_x = self.pca_node.execute(sfa_removed_x)

        if self.slow_feature_scaling_method == "data_dependent":
            if pca_output_dim > 0:
                self.magn_n_sfa_x = 1.0 * numx.median(
                    self.pca_node.d) ** 0.5
            else:
                self.magn_n_sfa_x = 1.0
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
            if verbose:
                print("method: data dependent")

        if self.pca_node.output_dim + self.num_sfa_features_preserved < \
                self.output_dim:
            er = "Error, the number of features computed is SMALLER than " \
                 + "the output dimensionality of the node: " + \
                 "self.pca_node.output_dim=" + str(self.pca_node.output_dim) \
                 + ", self.num_sfa_features_preserved=" + \
                 str(self.num_sfa_features_preserved) + ", self.output_dim=" \
                 + str(self.output_dim)
            raise TrainingException(er)

        # Finally, the output is the concatenation of scaled slow features
        # and remaining pca components
        sfa_pca_x = numx.concatenate((s_n_sfa_x, pca_x), axis=1)

        sfa_pca_x_truncated = sfa_pca_x[:, 0:self.output_dim]

        # Compute explained variance from amplitudes of output compared to
        # amplitudes of input. Only works because amplitudes of SFA are
        # scaled to be equal to explained variance, because PCA is
        # a rotation, and because data has zero mean
        self.evar = (sfa_pca_x_truncated ** 2).sum() / (x_zm ** 2).sum()
        if verbose:
            print("s_n_sfa_x:", s_n_sfa_x, "pca_x:", pca_x)
            print("sfa_pca_x_truncated:", sfa_pca_x_truncated, "x_zm:", x_zm)
            print("Variance(output) / Variance(input) is ", self.evar)
        self.stop_training()

    def multiple_train(self, x, block_size=None, train_mode=None,
                       node_weights=None, edge_weights=None, verbose=None):
        """This function should not be called directly. Use instead the train
        method, which will decide whether multiple-training is enabled, and
        call this function if needed.
        """
        self.input_dim = x.shape[1]
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print("Training iGSFANode (multiple train method)...")

        # Data mean is ignored by the multiple train method
        if self.x_mean is None:
            self.x_mean = numx.zeros(self.input_dim, dtype=self.dtype)
        x_zm = x

        # Reorder or pre-process the data before it is expanded, but only if
        # there is really an expansion.
        if self.pre_expansion_node_class and self.exp_node:
            er = "Unexpected parameters"
            raise TrainingException(er)
        else:
            x_pre_exp = x_zm

        if self.exp_node:
            if verbose:
                print("expanding x...")
            exp_x = self.exp_node.execute(x_pre_exp)
        else:
            exp_x = x_pre_exp

        self.expanded_dim = exp_x.shape[1]

        if self.max_length_slow_part is None:
            sfa_output_dim = min(self.expanded_dim, self.output_dim)
        else:
            sfa_output_dim = min(self.max_length_slow_part, self.expanded_dim,
                                 self.output_dim)

        if isinstance(self.delta_threshold, int):
            sfa_output_dim = min(sfa_output_dim, self.delta_threshold)
            sfa_output_dim = max(1, sfa_output_dim)

        # Apply SFA to expanded data
        if self.sfa_node is None:
            self.sfa_node = GSFANode(output_dim=sfa_output_dim,
                                     verbose=verbose)
        self.sfa_x_mean = 0
        self.sfa_x_std = 1.0

        self.sfa_node.train(exp_x, block_size=block_size,
                            train_mode=train_mode,
                            node_weights=node_weights,
                            edge_weights=edge_weights)

        if verbose:
            print("training PCA...")
        pca_output_dim = self.output_dim
        if self.pca_node is None:
            # If necessary, add reduce=True
            self.pca_node = mdp.nodes.PCANode(output_dim=pca_output_dim)
        sfa_removed_x = x
        self.pca_node.train(sfa_removed_x)

    def _stop_training(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if self.reconstruct_with_sfa or \
                (self.slow_feature_scaling_method not in [None,
                                                          "data_dependent"]):
            return
        # else, continue with multi-train method

        self.sfa_node.stop_training()
        if verbose:
            print("self.sfa_node.d", self.sfa_node.d)
        self.pca_node.stop_training()

        # Decide how many slow features are preserved
        if isinstance(self.delta_threshold, float):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = (self.sfa_node.d <=
                                               self.delta_threshold).sum()
        elif isinstance(self.delta_threshold, int):
            # here self.max_length_slow_part should be considered
            self.num_sfa_features_preserved = self.delta_threshold
            if self.delta_threshold > self.output_dim:
                er = "The provided integer delta_threshold " + \
                     "%d is larger than the output dimensionality %d" % \
                     (self.delta_threshold, self.output_dim)
                raise ValueError(er)
            if self.max_length_slow_part is not None and \
                            self.delta_threshold > self.max_length_slow_part:
                er = "The provided integer delta_threshold: %d" % \
                     self.delta_threshold + \
                     " is larger than max_length_slow_part: %d" % \
                     self.max_length_slow_part
                raise ValueError(er)
        else:
            ex = "Cannot handle type of self.delta_threshold:" + \
                 str(type(self.delta_threshold))
            raise ValueError(ex)

        if self.num_sfa_features_preserved > self.output_dim:
            self.num_sfa_features_preserved = self.output_dim

        SFANode_reduce_output_dim(self.sfa_node,
                                  self.num_sfa_features_preserved)
        if verbose:
            print("size of slow part:", self.num_sfa_features_preserved)

        final_pca_node_output_dim = self.output_dim - \
                                    self.num_sfa_features_preserved
        if final_pca_node_output_dim > self.pca_node.output_dim:
            er = "The number of features computed is SMALLER than the " + \
                 "output dimensionality of the node: " + \
                 "pca_node.output_dim=" + str(self.pca_node.output_dim) + \
                 ", num_sfa_features_preserved=" + \
                 str(self.num_sfa_features_preserved) + \
                 ", output_dim=" + str(self.output_dim)
            raise ValueError(er)
        PCANode_reduce_output_dim(self.pca_node, final_pca_node_output_dim,
                                  verbose=False)

        if verbose:
            print("self.pca_node.d", self.pca_node.d)
            print("ranking method...")
        if self.slow_feature_scaling_method is None:
            self.magn_n_sfa_x = 1.0
            if verbose:
                print("method: constant amplitude for all slow features")
        elif self.slow_feature_scaling_method == "data_dependent":
            # SFA components are given a variance equal to the median variance
            # of the principal components
            if self.pca_node.d.shape[0] > 0:
                self.magn_n_sfa_x = 1.0 * numx.median(self.pca_node.d) ** 0.5
            else:
                self.magn_n_sfa_x = 1.0
            if verbose:
                print("method: data dependent")
        else:
            er = "Unknown slow feature scaling method" + \
                 str(self.slow_feature_scaling_method)
            raise ValueError(er)
        self.evar = self.pca_node.explained_variance

    @staticmethod
    def _is_invertible():
        return True

    def _execute(self, x):
        """Extracts iGSFA features from some data. The node must have been
        already trained.
        """
        x_zm = x - self.x_mean

        if self.pre_expansion_node:
            x_pre_exp = self.pre_expansion_node.execute(x_zm)
        else:
            x_pre_exp = x_zm

        if self.exp_node:
            exp_x = self.exp_node.execute(x_pre_exp)
        else:
            exp_x = x_pre_exp

        sfa_x = self.sfa_node.execute(exp_x)

        n_sfa_x = (sfa_x - self.sfa_x_mean) / self.sfa_x_std

        if self.reconstruct_with_sfa:
            # approximate input linearly, done inline to preserve node
            x_pca_app = self.lr_node.execute(n_sfa_x)
            x_app = x_pca_app
        else:
            x_app = numx.zeros_like(x_zm)

        # Remove linear approximation from the centered data
        sfa_removed_x = x_zm - x_app

        # A method for feature scaling( +rotation)
        if self.reconstruct_with_sfa \
                and self.slow_feature_scaling_method == "QR_decomposition":
            s_n_sfa_x = numx.dot(n_sfa_x, self.R.T)
        # Another method for feature scaling (no rotation)
        elif self.reconstruct_with_sfa \
                and self.slow_feature_scaling_method == "sensitivity_based":
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
        elif self.slow_feature_scaling_method is None:
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
        elif self.slow_feature_scaling_method == "data_dependent":
            s_n_sfa_x = n_sfa_x * self.magn_n_sfa_x
        else:
            er = "unknown feature scaling method" + \
                 str(self.slow_feature_scaling_method)
            raise ValueError(er)

        # Apply PCA to sfa removed data
        if self.pca_node.output_dim > 0:
            pca_x = self.pca_node.execute(sfa_removed_x)
        else:
            # No reconstructive components present
            pca_x = numx.zeros((x.shape[0], 0))

        # Finally output is the concatenation of scaled slow features and
        # remaining pca components
        sfa_pca_x = numx.concatenate((s_n_sfa_x, pca_x), axis=1)

        return sfa_pca_x  # sfa_pca_x_truncated

    def _inverse(self, y, linear_inverse=True):
        """This method approximates an inverse function to the feature
        extraction.

        If linear_inverse is True, a linear method is used. Otherwise, a
        gradient-based non-linear method is used.
        """
        if linear_inverse:
            return self.linear_inverse(y)
        else:
            return self.nonlinear_inverse(y)

    def nonlinear_inverse(self, y, verbose=None):
        """Non-linear inverse approximation method.

        This method is experimental and should be used with care.
        Note: this function requires scipy.
        """
        if mdp.numx_description != 'scipy':
            raise NotImplementedError('This function requires scipy.')
        else:
            import scipy.optimize

        if verbose is None:
            verbose = self.verbose
        x_lin = self.linear_inverse(y)
        rmse_lin = ((y - self.execute(x_lin)) ** 2).sum(axis=1).mean() ** 0.5
        x_nl = numx.zeros_like(x_lin)
        y_dim = y.shape[1]
        x_dim = x_lin.shape[1]
        if y_dim < x_dim:
            num_zeros_filling = x_dim - y_dim
        else:
            num_zeros_filling = 0
        if verbose:
            print("x_dim=", x_dim, "y_dim=", y_dim, "num_zeros_filling=",
                  num_zeros_filling)
        y_long = numx.zeros(y_dim + num_zeros_filling)

        for i, y_i in enumerate(y):
            y_long[0:y_dim] = y_i
            if verbose:
                print("x_0=", x_lin[i])
                print("y_long=", y_long)
            plsq = scipy.optimize.leastsq(func=f_residual, x0=x_lin[i],
                                          args=(self, y_long),
                                          full_output=False)
            x_nl_i = plsq[0]
            if verbose:
                print("x_nl_i=", x_nl_i, "plsq[1]=", plsq[1])
            if plsq[1] != 2:
                print("Quitting: plsq[1]=", plsq[1])
            x_nl[i] = x_nl_i
            if verbose:
                y_i_app = self.execute(x_lin[i].reshape((1, -1)))
                print("|E_lin(%d)|=" % i,
                      ((y_i - y_i_app) ** 2).sum() ** 0.5)
                y_i_app = self.execute(x_nl_i.reshape((1, -1)))
                print("|E_nl(%d)|=" % i,
                      ((y_i - y_i_app) ** 2).sum() ** 0.5)
        rmse_nl = ((y - self.execute(x_nl)) ** 2).sum(axis=1).mean() ** 0.5
        if verbose:
            print("rmse_lin(all samples)=", rmse_lin,
                  "rmse_nl(all samples)=", rmse_nl)
        return x_nl

    def linear_inverse(self, y, verbose=True):
        """Linear inverse approximation method. """
        if verbose is None:
            verbose = self.verbose
        num_samples = y.shape[0]
        if y.shape[1] != self.output_dim:
            er = "Serious dimensionality inconsistency"
            raise TrainingException(er)

        sfa_pca_x_full = numx.zeros(
            (num_samples,
             self.pca_node.output_dim + self.num_sfa_features_preserved),
            dtype=self.dtype)
        sfa_pca_x_full[:, 0:self.output_dim] = y

        s_n_sfa_x = sfa_pca_x_full[:, 0:self.num_sfa_features_preserved]
        pca_x = sfa_pca_x_full[:, self.num_sfa_features_preserved:]

        if pca_x.shape[1] > 0:
            sfa_removed_x = self.pca_node.inverse(pca_x)
        else:
            sfa_removed_x = numx.zeros((num_samples, self.input_dim))

        # A method for feature scaling (+rotation)
        if self.reconstruct_with_sfa \
                and self.slow_feature_scaling_method == "QR_decomposition":
            n_sfa_x = numx.dot(s_n_sfa_x, self.Rpinv.T)
        else:
            n_sfa_x = s_n_sfa_x / self.magn_n_sfa_x

        # recall: sfa_x is n_sfa_x * self.sfa_x_std + self.sfa_x_mean
        if self.reconstruct_with_sfa:
            x_pca_app = self.lr_node.execute(n_sfa_x)
            x_app = x_pca_app
        else:
            x_app = numx.zeros_like(sfa_removed_x)

        x_zm = sfa_removed_x + x_app

        x = x_zm + self.x_mean

        if verbose:
            print("Data_std(x_zm)=", x_zm.var(axis=0))
            print("Data_std(x_app)=", x_app.var(axis=0))
            print("Data_std(sfa_removed_x)=", sfa_removed_x.var(axis=0))
        return x


def SFANode_reduce_output_dim(sfa_node, new_output_dim, verbose=False):
    """ This function modifies an already trained SFA node (or GSFA node),
    reducing the number of preserved SFA features to new_output_dim features.
    The modification is done in place
    """
    if verbose:
        print("Updating the output dimensionality of SFA node")
    if new_output_dim > sfa_node.output_dim:
        er = "Cannot increase the output dimensionality of the SFA node"
        raise ValueError(er)
    sfa_node.d = sfa_node.d[:new_output_dim]
    sfa_node.sf = sfa_node.sf[:, :new_output_dim]
    sfa_node._bias = sfa_node._bias[:new_output_dim]
    sfa_node._output_dim = new_output_dim


def PCANode_reduce_output_dim(pca_node, new_output_dim, verbose=False):
    """ This function modifies an already trained PCA node,
    reducing the number of preserved SFA features to new_output_dim features.
    The modification is done in place. Also the explained variance field is
    updated.
    """
    if verbose:
        print("Updating the output dimensionality of PCA node")
    if new_output_dim > pca_node.output_dim:
        er = "Cannot increase the output dimensionality of the PCA node"
        raise ValueError(er)

    original_total_variance = pca_node.d.sum()
    original_explained_variance = pca_node.explained_variance
    pca_node.d = pca_node.d[0:new_output_dim]
    pca_node.v = pca_node.v[:, 0:new_output_dim]
    # pca_node.avg is not affected by this method!
    pca_node._output_dim = new_output_dim
    pca_node.explained_variance = \
        original_explained_variance * \
        pca_node.d.sum() / original_total_variance


# Computes output errors dimension by dimension for a single sample:
# y - node.execute(x_app)
# The library fails when dim(x_app) > dim(y), thus filling of x_app with
# zeros is recommended
def f_residual(x_app_i, node, y_i):
    res_long = numx.zeros_like(y_i)
    y_i = y_i.reshape((1, -1))
    y_i_short = y_i[:, 0:node.output_dim]
    res = (y_i_short - node.execute(x_app_i.reshape((1, -1)))).flatten()
    res_long[0:len(res)] = res
    return res_long


###############################################################################
#   EXAMPLES THAT SHOW HOW GSFA CAN BE USED                                   #
###############################################################################

def example_clustered_graph():
    print("\n****************************************************************")
    print("*Example of training GSFA using a clustered graph")
    cluster_size = 20
    num_clusters = 5
    num_samples = cluster_size * num_clusters
    dim = 20
    output_dim = 2
    x = numx.random.normal(size=(num_samples, dim))
    x += 0.1 * numx.arange(num_samples).reshape((num_samples, 1))

    print("x=", x)

    GSFA_n = GSFANode(output_dim=output_dim)

    def identity(xx):
        return xx

    def norm2(xx):  # Computes the norm of each sample returning an Nx1 array
        return ((xx ** 2).sum(axis=1) ** 0.5).reshape((-1, 1))

    Exp_n = mdp.nodes.GeneralExpansionNode([identity, norm2])

    exp_x = Exp_n.execute(x)  # Expanded data
    GSFA_n.train(exp_x, train_mode="clustered", block_size=cluster_size)
    GSFA_n.stop_training()

    print("GSFA_n.d=", GSFA_n.d)

    y = GSFA_n.execute(Exp_n.execute(x))
    print("y", y)
    print("Standard delta values of output features y:", comp_delta(y))

    x_test = numx.random.normal(size=(num_samples, dim))
    x_test += 0.1 * numx.arange(num_samples).reshape((num_samples, 1))
    y_test = GSFA_n.execute(Exp_n.execute(x_test))
    print("y_test", y_test)
    print("Standard delta values of output features y_test:",
          comp_delta(y_test))


def example_pathological_outputs(experiment):
    print("\n ***************************************************************")
    print("*Pathological responses.",
          "Experiment on graph with weakly connected samples")
    x = numx.random.normal(size=(20, 19))
    x2 = numx.random.normal(size=(20, 19))

    l = numx.random.normal(size=20)
    l -= l.mean()
    l /= l.std()
    l.sort()

    half_width = 3

    v = numx.ones(20)
    e = {}
    for t in range(19):
        e[(t, t + 1)] = 1.0
    e[(0, 0)] = 0.5
    e[(19, 19)] = 0.5
    train_mode = "graph"

    # Select the experiment to perform, from 0 to 11
    print("experiment", experiment)
    if experiment == 0:
        exp_title = "Original linear SFA graph. Experiment 0"
    elif experiment == 1:
        v[0] = 10.0
        v[10] = 0.1
        v[19] = 10.0
        exp_title = "Modified node weights. Experiment 1"
    elif experiment == 2:
        v[0] = 10.0
        v[19] = 0.1
        exp_title = "Modified node weights. Experiment 2"
    elif experiment == 3:
        e[(0, 1)] = 0.1
        e[(18, 19)] = 10.0
        exp_title = "Modified edge weights. Experiment 3"
    elif experiment == 4:
        e[(0, 1)] = 0.01
        e[(18, 19)] = 0.01
        e[(15, 17)] = 0.5
        e[(16, 18)] = 0.5
        e[(12, 14)] = 0.5
        e[(3, 5)] = 0.5
        e[(4, 6)] = 0.5
        e[(5, 7)] = 0.5

        exp_title = "Modified edge weights. Experiment 4"
    elif experiment == 5:
        e[(10, 11)] = 0.02
        e[(1, 2)] = 0.02
        e[(3, 5)] = 1.0
        e[(7, 9)] = 1.0
        e[(17, 19)] = 1.0
        e[(14, 16)] = 1.0

        exp_title = "Modified edge weights. Experiment 5"
    elif experiment == 6:
        e[(6, 7)] = 0.1
        e[(5, 6)] = 0.1
        exp_title = "Modified edge weights. Experiment 6"
    elif experiment == 7:
        e = {}
        for j1 in range(19):
            for j2 in range(j1 + 1, 20):
                e[(j1, j2)] = 1 / (l[j2] - l[j1] + 0.00005)
        exp_title = "Modified edge weights for labels with " + \
                    "w12 = 1/(l2-l1+0.00005). Experiment 7"
    elif experiment == 8:
        e = {}
        for j1 in range(19):
            for j2 in range(j1 + 1, 20):
                e[(j1, j2)] = numx.exp(-0.25 * (l[j2] - l[j1]) ** 2)
        exp_title = "Modified edge weights for labels with " + \
                    "w12 = exp(-0.25*(l2-l1)**2). Experiment 8"
    elif experiment == 9:
        e = {}
        for j1 in range(19):
            for j2 in range(j1 + 1, 20):
                if l[j2] - l[j1] < 1.5:
                    e[(j1, j2)] = 1 / (l[j2] - l[j1] + 0.0005)
        exp_title = "Modified edge weights w12 = 1/(l2-l1+0.0005), " + \
                    "for l2-l1<1.5. Experiment 9"
    elif experiment == 10:
        exp_title = "Mirroring training graph, w=%d" % half_width + \
                    ". Experiment 10"
        train_mode = "smirror_window%d" % half_width
        e = {}
    elif experiment == 11:
        exp_title = "Node weight adjustment training graph, w=%d " % \
                    half_width + ". Experiment 11"
        train_mode = "window%d" % half_width
        e = {}
    else:
        er = "Unknown experiment: " + str(experiment)
        raise ValueError(er)

    n = GSFANode(output_dim=5)
    if experiment in (10, 11):
        n.train(x, train_mode=train_mode)
    else:
        n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    print("/" * 20, "Brute delta values of GSFA features (training/test):")
    y = n.execute(x)
    y2 = n.execute(x2)
    if e != {}:
        print(graph_delta_values(y, e))
        print(graph_delta_values(y2, e))

    D = numx.zeros(20)
    for (j1, j2) in e:
        D[j1] += e[(j1, j2)] / 2.0
        D[j2] += e[(j1, j2)] / 2.0

    import matplotlib as mpl
    mpl.use('Qt4Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Outputs overfitted to training data, node weights=" + str(v))
    plt.xlabel(exp_title + "\n With D (half the sum of all edges " +
               "from/to each vertex)=" + str(D))
    plt.xticks(numx.arange(0, 20, 1))
    plt.plot(y)
    if experiment in (6, 7, 8):
        if y[0, 0] > 0:
            l *= -1
        plt.plot(l, "*")
    plt.show()


def example_continuous_edge_weights():
    print("\n****************************************************************")
    print("*Testing continuous edge weigths w_{n,n'} = 1/(|l_n'-l_n|+k)")
    x = numx.random.normal(size=(20, 19))
    x2 = numx.random.normal(size=(20, 19))

    l = numx.random.normal(size=20)
    l -= l.mean()
    l /= l.std()
    l.sort()
    k = 0.0001

    v = numx.ones(20)
    e = {}
    for n1 in range(20):
        for n2 in range(20):
            if n1 != n2:
                e[(n1, n2)] = 1.0 / (numx.absolute(l[n2] - l[n1]) + k)

    exp_title = "Original linear SFA graph"
    n = GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    print("/" * 20, "Brute delta values of GSFA features (training/test):")
    y = n.execute(x)
    y2 = n.execute(x2)
    if e != {}:
        print(graph_delta_values(y, e))
        print(graph_delta_values(y2, e))

    D = numx.zeros(20)
    for (j1, j2) in e:
        D[j1] += e[(j1, j2)] / 2.0
        D[j2] += e[(j1, j2)] / 2.0

    import matplotlib as mpl
    mpl.use('Qt4Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Overfitted outputs on training data,v=" + str(v))
    plt.xlabel(exp_title + "\n With D=" + str(D))
    plt.xticks(numx.arange(0, 20, 1))
    plt.plot(y)
    plt.plot(l, "*")
    plt.show()


###############################################################################
#   AN EXAMPLE OF HOW iGSFA CAN BE USED                                       #
###############################################################################

def example_iGSFA():
    print("\n\n**************************************************************")
    print("*Example of training iGSFA on random data")
    num_samples = 1000
    dim = 20
    verbose = False
    x = numx.random.normal(size=(num_samples, dim))
    x[:, 0] += 2.0 * numx.arange(num_samples) / num_samples
    x[:, 1] += 1.0 * numx.arange(num_samples) / num_samples
    x[:, 2] += 0.5 * numx.arange(num_samples) / num_samples

    x_test = numx.random.normal(size=(num_samples, dim))
    x_test[:, 0] += 2.0 * numx.arange(num_samples) / num_samples
    x_test[:, 1] += 1.0 * numx.arange(num_samples) / num_samples
    x_test[:, 2] += 0.5 * numx.arange(num_samples) / num_samples

    def zero_mean_unit_var(x):
        x -= x.mean(axis=0)
        x /= x.std(axis=0)
        return x

    print("Node creation and training")
    n = iGSFANode(output_dim=15, reconstruct_with_sfa=False,
                  slow_feature_scaling_method="data_dependent",
                  verbose=verbose)
    n.train(x, train_mode="regular")
    n.stop_training()

    y = n.execute(x)
    y_test = n.execute(x_test)

    print("y=", y)
    print("y_test=", y_test)
    print("Standard delta values of output features y:", comp_delta(y))
    print("Standard delta values of output features y_test:",
          comp_delta(y_test))
    y_norm = zero_mean_unit_var(y)
    y_test_norm = zero_mean_unit_var(y_test)
    print("Standard delta values of output features y after constraint ",
          "enforcement:", comp_delta(y_norm))
    print("Standard delta values of output features y_test after constraint ",
          "enforcement:", comp_delta(y_test_norm))

    x_app_lin = n.inverse(y)
    x_app_nonlin = n.nonlinear_inverse(y)
    y_app_lin = n.execute(x_app_lin)
    y_app_nonlin = n.execute(x_app_nonlin)
    print("|y - y_app_lin|", ((y - y_app_lin)**2).mean())
    print("|y - y_app_nonlin|", ((y - y_app_nonlin)**2).mean())

if __name__ == "__main__":
    for experiment_number in range(0, 12):
        example_pathological_outputs(experiment=experiment_number)
    example_continuous_edge_weights()
    example_clustered_graph()
    example_iGSFA()
