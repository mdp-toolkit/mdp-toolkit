#####################################################################################################################
# test_iGSFANode: Tests for the Information-Preserving Graph-Based SFA Node (iGSFANode) as defined by                #
#                 the Cuicuilco framework                                                                           #
#                                                                                                                   #
# By Alberto-N Escalante-B. Alberto.Escalante@ini.rub.de                                                            #
# Ruhr-University Bochum, Institute for Neural Computation, Group of Prof. Dr. Wiskott                              #
#####################################################################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy
import copy

import pytest
import mdp


import cuicuilco.patch_mdp  # Is this necessary???
from cuicuilco.gsfa_nodes import comp_delta, GSFANode, iGSFANode, SFANode_reduce_output_dim, PCANode_reduce_output_dim

# TODO: rename offsetting_mode -> slow_feature_scaling_method
#       *test_SFANode_reduce_output_dim (extraction and inverse)
#       *test_PCANode_reduce_output_dim (extraction and inverse)

#iGSFANode(input_dim=None, output_dim=None, pre_expansion_node_class=None, pre_expansion_out_dim=None,
#                 expansion_funcs=None, expansion_output_dim=None, expansion_starting_point=None,
#                 max_length_slow_part=None, slow_feature_scaling_method="sensitivity_based_pure", delta_threshold=1.9999,
#                 reconstruct_with_sfa=True, **argv)


def test_automatic_stop_training():
    """ Test that verifies that iGSFA automatically calls stop training when trained on single batch mode
    """
    x = numpy.random.normal(size=(300, 15))

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, slow_feature_scaling_method=None)
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, slow_feature_scaling_method="data_dependent")
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, slow_feature_scaling_method="sensitivity_based")
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")

    n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, slow_feature_scaling_method="QR_decomposition")
    n.train(x, train_mode="regular")
    with pytest.raises(mdp.TrainingFinishedException):
        n.train(x, train_mode="regular")


def test_no_automatic_stop_training():
    """ Test that verifies that iGSFA does not call stop training when when multiple-train is used
    """
    x = numpy.random.normal(size=(300, 15))
    n = iGSFANode(output_dim=5, reconstruct_with_sfa=False, slow_feature_scaling_method=None)
    n.train(x, train_mode="regular")
    n.train(x, train_mode="regular")
    n.stop_training()

    n = iGSFANode(output_dim=5, reconstruct_with_sfa=False, slow_feature_scaling_method="data_dependent")
    n.train(x, train_mode="regular")
    n.train(x, train_mode="regular")
    n.stop_training()


def test_slow_feature_scaling_methods():
    """ Test that executes each feature scaling method and verifies that (most of them) only change the
    scale of the slow features in the slow part but do not mix them.
    """
    x = numpy.random.normal(size=(300, 15))

    all_slow_feature_scaling_methods = ["QR_decomposition", "sensitivity_based", None, "data_dependent"]
    num_slow_feature_scaling_methods = len(all_slow_feature_scaling_methods)
    output_features = []
    for slow_feature_scaling_method in all_slow_feature_scaling_methods:
        n = iGSFANode(output_dim=15, reconstruct_with_sfa=True, slow_feature_scaling_method=slow_feature_scaling_method)
        n.train(x, train_mode="regular")
        if n.is_training():
            n.stop_training()
        output_features.append(n.execute(x))


    size_slow_part = n.sfa_node.output_dim
    print("size_slow_part:", size_slow_part)
    for i in range(num_slow_feature_scaling_methods):
        output_features[i] = output_features[i][:,:size_slow_part]
    first_sample_y_data_dependent = output_features[num_slow_feature_scaling_methods-1][0]
    for i in range(1, len(all_slow_feature_scaling_methods)-1):
        print("checking feature equivalence between", all_slow_feature_scaling_methods[i], "and",
              all_slow_feature_scaling_methods[num_slow_feature_scaling_methods-1])
        first_sample_y_i = output_features[i][0]
        y = output_features[i] * first_sample_y_data_dependent / first_sample_y_i
        assert (y - output_features[num_slow_feature_scaling_methods-1]) == pytest.approx(0.0)


def test_enforce_int_delta_threshold_le_output_dim():
    x = numpy.random.normal(size=(300, 15))
    # No automatic stop_training
    n = iGSFANode(output_dim=5, reconstruct_with_sfa=False, slow_feature_scaling_method=None, delta_threshold=6)
    n.train(x, train_mode="regular")
    n.train(x**3, train_mode="regular")
    with pytest.raises(Exception):
        n.stop_training()
    # Automatic stop_training
    n = iGSFANode(output_dim=5, reconstruct_with_sfa=True, slow_feature_scaling_method=None, delta_threshold=6)
    with pytest.raises(Exception):
        n.train(x, train_mode="regular")


def test_enforce_int_delta_threshold_le_max_length_slow_part():
    x = numpy.random.normal(size=(300, 10))
    # No automatic stop_training
    n = iGSFANode(output_dim=8, reconstruct_with_sfa=False, slow_feature_scaling_method=None,
                  max_length_slow_part=5, delta_threshold=6)
    n.train(x, train_mode="regular")
    n.train(x**3, train_mode="regular")
    with pytest.raises(Exception):
        n.stop_training()
    # Automatic stop_training
    n = iGSFANode(output_dim=8, reconstruct_with_sfa=True, slow_feature_scaling_method=None,
                  max_length_slow_part=5, delta_threshold=6)
    with pytest.raises(Exception):
        n.train(x, train_mode="regular")


def test_SFANode_reduce_output_dim():
    x = numpy.random.normal(size=(300, 15))
    n = mdp.nodes.SFANode(output_dim=10)
    n.train(x)
    n.stop_training()
    y1 = n.execute(x)[:, 0:6]

    n2 = copy.deepcopy(n)
    SFANode_reduce_output_dim(n2, 6)
    y2 = n2.execute(x)
    assert (y2 - y1) == pytest.approx(0.0)


def test_PCANode_reduce_output_dim():
    x = numpy.random.normal(size=(300, 15))
    n = mdp.nodes.PCANode(output_dim=10)
    n.train(x)
    n.stop_training()
    y1 = n.execute(x)[:, 0:6]

    n2 = copy.deepcopy(n)
    PCANode_reduce_output_dim(n2, 6)
    y2 = n2.execute(x)
    assert (y2 - y1) == pytest.approx(0.0)


def test_equivalence_GSFA_iGSFA_for_DT_4_0():
    """ Test of iGSFA and GSFA when delta_threshold is larger than 4.0
    """
    x = numpy.random.normal(size=(300, 15))

    n = iGSFANode(output_dim=5, slow_feature_scaling_method=None, delta_threshold=4.10)
    n.train(x, train_mode="regular")
    # n.stop_training() has been automatically called

    y = n.execute(x)
    deltas_igsfa = comp_delta(y)

    n2 = GSFANode(output_dim=5)
    n2.train(x, train_mode="regular")
    n2.stop_training()

    y2 = n2.execute(x)
    deltas_gsfa = comp_delta(y2)
    assert (deltas_igsfa - deltas_gsfa) == pytest.approx(0.0)


def test_equivalence_GSFA_PCA_for_DT_0():
    """ Test of iGSFA and PCA when delta_threshold is smaller than 0.0
    """
    x = numpy.random.normal(size=(300, 15))

    n = iGSFANode(output_dim=5, slow_feature_scaling_method=None, delta_threshold=0.0)
    n.train(x, train_mode="regular")
    # n.stop_training() has been automatically called

    y = n.execute(x)
    deltas_igsfa = comp_delta(y)

    n2 = mdp.nodes.PCANode(output_dim=5)
    n2.train(x)
    n2.stop_training()

    y2 = n2.execute(x)
    deltas_pca = comp_delta(y2)
    assert (deltas_igsfa - deltas_pca) == pytest.approx(0.0)
