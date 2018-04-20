#####################################################################################################################
# test_GSFANode: Tests for the Graph-Based SFA Node (GSFANode) as defined by the Cuicuilco framework                #
#                                                                                                                   #
# By Alberto Escalante. Alberto.Escalante@ini.rub.de                                                                #
# Ruhr-University Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from past.utils import old_div
from ._tools import *

# import numx
import pytest
# import mdp

from mdp.nodes.gsfa_nodes import graph_delta_values, comp_delta

#TODO: test invalid parameters (training mode, block size, etc)


def test_equivalence_SFA_GSFA_regular_mode():
    """ Tests the equivalence of Standard SFA and GSFA when trained using the "regular" training mode
    """
    num_samples = 200
    correction_factor_scale = ((num_samples - 1.0) / num_samples) ** 0.5
    input_dim = 15
    x = numx.random.normal(size=(num_samples, input_dim))
    x2 = numx.random.normal(size=(num_samples, input_dim))

    print("Training GSFA (regular mode):")
    output_dim = 5
    n = mdp.nodes.GSFANode(output_dim=output_dim)
    n.train(x, train_mode="regular")
    n.stop_training()

    print("Training SFA:")
    n_sfa = mdp.nodes.SFANode(output_dim=output_dim)
    n_sfa.train(x)
    n_sfa.stop_training()

    y = n.execute(x)
    y *= correction_factor_scale
    assert y.shape[1] == output_dim, "Output dimensionality %d was supposed to be %d" % (y.shape[1], output_dim)
    print("y[0]:", y[0])
    print("y.mean:", y.mean(axis=0))
    print("y.var:", (y**2).mean(axis=0))
    y2 = n.execute(x2)
    y2 *= correction_factor_scale

    y_sfa = n_sfa.execute(x)
    print("y_sfa[0]:", y_sfa[0])
    print("y_sfa.mean:", y_sfa.mean(axis=0))
    print("y_sfa.var:", (y_sfa**2).mean(axis=0))
    y2_sfa = n_sfa.execute(x2)

    signs_sfa = numx.sign(y_sfa[0,:])
    signs_gsfa = numx.sign(y[0,:])
    y = y * signs_gsfa * signs_sfa
    y2 = y2 * signs_gsfa * signs_sfa

    print("y_sfa:", y_sfa, "y:", y)
    assert (y_sfa - y) == pytest.approx(0.0)
    assert (y2_sfa - y2) == pytest.approx(0.0)


def test_equivalence_GSFA_clustered_and_classification_modes():
    """ Tests the equivalence of GSFA when trained using the "clustered" and "classification" training modes.

    Notice that the clustered mode assumes the samples with the same labels are contiguous.
    """
    num_classes = 5
    num_samples_per_class = numx.array([numx.random.randint(10, 30) for j in range(num_classes)])
    num_samples = num_samples_per_class.sum()
    classes = []
    for i, j in enumerate(num_samples_per_class):
        classes += [i] * j
    classes = numx.array(classes)
    print("classes:", classes)
    input_dim = 15
    x = numx.random.normal(size=(num_samples, input_dim))

    print("Training GSFA (clustered mode):")
    output_dim = num_classes - 1
    n = mdp.nodes.GSFANode(output_dim=output_dim)
    n.train(x, train_mode="clustered", block_size=num_samples_per_class)
    n.stop_training()


    sorting = numx.arange(num_samples)
    numx.random.shuffle(sorting)
    x_s = x[sorting]
    classes_s = classes[sorting]
    print("Training GSFA (classification mode):")
    n2 = mdp.nodes.GSFANode(output_dim=output_dim)
    n2.train(x_s, train_mode=("classification", classes_s, 1.0))
    n2.stop_training()

    y_clustered = n.execute(x)
    print("reordering outputs of clustered mode")
    y_clustered = y_clustered[sorting]
    print("y_clustered[0]:", y_clustered[0])
    print("y_clustered.mean:", y_clustered.mean(axis=0))
    print("y_clustered.var:", (y_clustered**2).mean(axis=0))

    y_classification = n2.execute(x_s)
    print("y_classification[0]:", y_classification[0])
    print("y_classification.mean:", y_classification.mean(axis=0))
    print("y_classification.var:", (y_classification**2).mean(axis=0))


    signs_gsfa_classification = numx.sign(y_classification[0,:])
    signs_gsfa_clustered = numx.sign(y_clustered[0,:])
    y_clustered = y_clustered * signs_gsfa_clustered * signs_gsfa_classification

    assert (y_clustered - y_classification) == pytest.approx(0.0)


def test_GSFA_zero_mean_unit_variance_graph():
    """ Test of GSFA for zero-mean unit variance constraints on random data and graph, edge dictionary mode
    """
    x = numx.random.normal(size=(200, 15))
    v = numx.ones(200)
    e = {}
    for i in range(1500):
        n1 = numx.random.randint(200)
        n2 = numx.random.randint(200)
        e[(n1, n2)] = numx.random.normal() + 1.0
    n = mdp.nodes.GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    y = n.execute(x)
    assert y.mean(axis=0) == pytest.approx(0.0)
    assert (y**2).mean(axis=0) == pytest.approx(1.0)


def test_basic_GSFA_edge_dict():
    """ Basic test of GSFA on random data and graph, edge dictionary mode
    """
    x = numx.random.normal(size=(200, 15))
    v = numx.ones(200)
    e = {}
    for i in range(1500):
        n1 = numx.random.randint(200)
        n2 = numx.random.randint(200)
        e[(n1, n2)] = numx.random.normal() + 1.0
    n = mdp.nodes.GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    y = n.execute(x)
    delta_values_training_data = graph_delta_values(y, e)

    x2 = numx.random.normal(size=(200, 15))
    y2 = n.execute(x2)
    y2 = y2 - y2.mean(axis=0)  # enforce zero mean
    y2 /= ((y2**2).mean(axis=0) ** 0.5)  # enforce zero-mean
    # print("y2 means:", y2.mean(axis=0))
    # print("y2 std:", (y2**2).mean(axis=0))

    delta_values_test_data = graph_delta_values(y2, e)
    assert (delta_values_training_data < delta_values_test_data).all()
    # print("Graph delta values of training data", graph_delta_values(y, e))
    # print("Graph delta values of test data (should be larger than for training)", graph_delta_values(y2, e))


def test_equivalence_SFA_GSFA_linear_graph():
    """ Tests the equivalence of Standard SFA and GSFA when trained using an appropriate linear graph (graph mode)
    """
    num_samples = 200
    correction_factor_scale = ((num_samples - 1.0) / num_samples) ** 0.5
    input_dim = 15
    x = numx.random.normal(size=(num_samples, input_dim))
    x2 = numx.random.normal(size=(num_samples, input_dim))

    v = numx.ones(num_samples)
    e = {}
    for t in range(num_samples - 1):
        e[(t, t + 1)] = 1.0
    e[(0, 0)] = 0.5
    e[(num_samples - 1, num_samples - 1)] = 0.5

    print("Training GSFA:")
    output_dim = 5
    n = mdp.nodes.GSFANode(output_dim=output_dim)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()

    print("Training SFA:")
    n_sfa = mdp.nodes.SFANode(output_dim=output_dim)
    n_sfa.train(x)
    n_sfa.stop_training()

    y = n.execute(x)
    y *= correction_factor_scale
    assert y.shape[1] == output_dim, "Output dimensionality %d was supposed to be %d" % (y.shape[1], output_dim)
    print("y[0]:", y[0])
    print("y.mean:", y.mean(axis=0))
    print("y.var:", (y**2).mean(axis=0))
    y2 = n.execute(x2)
    y2 *= correction_factor_scale

    y_sfa = n_sfa.execute(x)
    print("y_sfa[0]:", y_sfa[0])
    print("y_sfa.mean:", y_sfa.mean(axis=0))
    print("y_sfa.var:", (y_sfa**2).mean(axis=0))
    y2_sfa = n_sfa.execute(x2)

    signs_sfa = numx.sign(y_sfa[0,:])
    signs_gsfa = numx.sign(y[0,:])
    y = y * signs_gsfa * signs_sfa
    y2 = y2 * signs_gsfa * signs_sfa

    assert (y_sfa - y) == pytest.approx(0.0)
    assert (y2_sfa - y2) == pytest.approx(0.0)






# FUTURE: Is it worth it to have so many methods? I guess the mirroring windows are enough, they have constant
# node weights and the edge weights almost fulfill consistency
def test_equivalence_window3_fwindow3():
    """Tests the equivalence of slow and fast mirroring sliding windows for GSFA
    """
    x = numx.random.normal(size=(200, 15))
    training_modes = ("window3", "fwindow3")

    delta_values = []
    for training_mode in training_modes:
        n = mdp.nodes.GSFANode(output_dim=5)
        n.train(x, train_mode=training_mode)
        n.stop_training()

        y = n.execute(x)
        delta = comp_delta(y)
        # print("**Brute Delta Values of mode %s are: " % training_mode, delta)
        delta_values.append(delta)

    # print(delta_values)
    assert (delta_values[1] - delta_values[0]) == pytest.approx(0.0)


def test_equivalence_smirror_window3_mirror_window3():
    """Tests the equivalence of slow and fast mirroring sliding windows for GSFA
    """
    x = numx.random.normal(size=(200, 15))
    training_modes = ("smirror_window3", "mirror_window3")

    delta_values = []
    for training_mode in training_modes:
        n = mdp.nodes.GSFANode(output_dim=5)
        n.train(x, train_mode=training_mode)
        n.stop_training()

        y = n.execute(x)
        delta = comp_delta(y)
        # print("**Brute Delta Values of mode %s are: " % training_mode, delta)
        delta_values.append(delta)

    # print(delta_values)
    assert (delta_values[1] - delta_values[0]) == pytest.approx(0.0)


def test_equivalence_smirror_window32_mirror_window32():
    """Tests the equivalence of slow and fast mirroring sliding windows for GSFA
    """
    x = numx.random.normal(size=(200, 15))
    training_modes = ("smirror_window32", "mirror_window32")

    delta_values = []
    for training_mode in training_modes:
        n = mdp.nodes.GSFANode(output_dim=5)
        n.train(x, train_mode=training_mode)
        n.stop_training()

        y = n.execute(x)
        delta = comp_delta(y)
        # print("**Brute Delta Values of mode %s are: " % training_mode, delta)
        delta_values.append(delta)

    # print(delta_values)
    assert (delta_values[1] - delta_values[0]) == pytest.approx(0.0)


def test_equivalence_update_graph_and_update_graph_old():
    """ Basic test of GSFA on random data and graph, edge dictionary mode
    """
    x = numx.random.normal(size=(200, 15))
    v = numx.ones(200)
    e = {}
    for i in range(1500):
        n1 = numx.random.randint(200)
        n2 = numx.random.randint(200)
        e[(n1, n2)] = numx.random.normal() + 1.0
    n = mdp.nodes.GSFANode(output_dim=5)
    n.train(x, train_mode="graph", node_weights=v, edge_weights=e)
    n.stop_training()
    y = n.execute(x)

    n2 = mdp.nodes.GSFANode(output_dim=5)
    n2.train(x, train_mode="graph_old", node_weights=v, edge_weights=e)
    n2.stop_training()
    y2 = n2.execute(x)

    assert (y - y2) == pytest.approx(0.0)

