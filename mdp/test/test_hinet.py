import pytest
import io
import mdp.hinet as mh
from ._tools import *
from future import standard_library
from builtins import zip
from builtins import range

standard_library.install_aliases()

"""These are test functions for hinet.
"""


def _get_new_flow():
    return mdp.Flow([mdp.nodes.NoiseNode(), mdp.nodes.SFANode()])


def _get_new_nodes():
    return [mdp.nodes.CuBICANode(input_dim=1, whitened=True), mdp.nodes.CuBICANode(input_dim=2, whitened=True),
            mdp.nodes.CuBICANode(input_dim=1, whitened=True), ]


def _get_sigle_node():
    return mdp.nodes.CuBICANode(input_dim=2, whitened=True)


NODES = [(mh.FlowNode, [_get_new_flow], None),
         (mh.Layer, [_get_new_nodes], None),
         (mh.CloneLayer, [_get_sigle_node, 2], None),
         ]


def test_flownode_training():
    flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2),
                     mdp.nodes.PCANode(output_dim=15, reduce=True),
                     mdp.nodes.PolynomialExpansionNode(degree=2),
                     mdp.nodes.PCANode(output_dim=3, reduce=True)])
    flownode = mh.FlowNode(flow)
    x = numx_rand.random([300, 20])
    while flownode.get_remaining_train_phase() > 0:
        flownode.train(x)
        flownode.stop_training()
    flownode.execute(x)


def test_flownode_trainability():
    flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2)])
    flownode = mh.FlowNode(flow)
    assert flownode.is_trainable() is False
    flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2),
                     mdp.nodes.PCANode(output_dim=15),
                     mdp.nodes.PolynomialExpansionNode(degree=2),
                     mdp.nodes.PCANode(output_dim=3)])
    flownode = mh.FlowNode(flow)
    assert flownode.is_trainable() is True


def test_flownode_invertibility():
    flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2)])
    flownode = mh.FlowNode(flow)
    assert flownode.is_invertible() is False
    flow = mdp.Flow([mdp.nodes.PCANode(output_dim=15),
                     mdp.nodes.SFANode(),
                     mdp.nodes.PCANode(output_dim=3)])
    flownode = mh.FlowNode(flow)
    assert flownode.is_invertible() is True


def test_flownode_pretrained_node():
    x = numx_rand.random([100, 10])
    pretrained_node = mdp.nodes.PCANode(output_dim=6)
    pretrained_node.train(x)
    pretrained_node.stop_training()
    flow = mdp.Flow([pretrained_node,
                     mdp.nodes.PolynomialExpansionNode(degree=2),
                     mdp.nodes.PCANode(output_dim=3)])
    flownode = mh.FlowNode(flow)
    while flownode.get_remaining_train_phase() > 0:
        flownode.train(x)
        flownode.stop_training()
    flownode.execute(x)


def test_flownode_fix_nodes_dimensions1():
    x = numx_rand.random([100, 10])
    last_node = mdp.nodes.IdentityNode()
    flow = mdp.Flow([mdp.nodes.PCANode(output_dim=3),
                     mdp.nodes.IdentityNode(),
                     last_node])
    flownode = mh.FlowNode(flow)
    flownode.train(x)
    flownode.stop_training()
    # check that the dimensions of NoiseNode and FlowNode where all set
    # by calling _fix_nodes_dimensions
    assert flownode.output_dim == 3
    assert last_node.input_dim == 3
    assert last_node.output_dim == 3


def test_flownode_fix_nodes_dimensions2():
    flow = mdp.Flow([mdp.nodes.IdentityNode(),
                     mdp.nodes.IdentityNode()])
    flownode = mh.FlowNode(flow)
    # this should fail, since the internal nodes don't have fixed dims
    pytest.raises(mdp.InconsistentDimException, lambda: flownode.set_output_dim(10))
    x = numx_rand.random([100, 10])
    flownode.execute(x)
    assert flownode.output_dim == 10


def test_flownode_fix_nodes_dimensions3():
    flow = mdp.Flow([mdp.nodes.IdentityNode()])
    flownode = mh.FlowNode(flow)
    # for a single node this should not raise an Exception
    flownode.set_output_dim(10)
    x = numx_rand.random([100, 10])
    flownode.execute(x)


def test_flownode_pretrained_flow():
    flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2),
                     mdp.nodes.PCANode(output_dim=15, reduce=True),
                     mdp.nodes.PolynomialExpansionNode(degree=2),
                     mdp.nodes.PCANode(output_dim=3, reduce=True)])
    flownode = mh.FlowNode(flow)
    x = numx_rand.random([300, 20])
    while flownode.get_remaining_train_phase() > 0:
        flownode.train(x)
        flownode.stop_training()
    # build new flownode with the trained nodes
    flownode = mh.FlowNode(flow)
    assert not flownode.is_training()
    flownode.execute(x)


def test_flownode_copy1():
    flow = mdp.Flow([mdp.nodes.PCANode(), mdp.nodes.SFANode()])
    flownode = mh.FlowNode(flow)
    flownode.copy()


def test_flownode_copy2():
    # Test that the FlowNode copy method delegates to internal nodes.
    class CopyFailException(Exception):
        pass

    class CopyFailNode(mdp.Node):
        def _train(self, x):
            pass

        def copy(self, protocol=None):
            raise CopyFailException()

    flow = mdp.Flow([mdp.Node(), CopyFailNode()])
    flownode = mh.FlowNode(flow)
    pytest.raises(CopyFailException, flownode.copy)


def _pca_nodes(input_dims, output_dims):
    return [mdp.nodes.PCANode(input_dim=ind, output_dim=outd)
            for ind, outd in zip(input_dims, output_dims)]


def test_layer():
    layer = mh.Layer(_pca_nodes([10, 17, 3], [5, 3, 1]))
    x = numx_rand.random([100, 30]).astype('f')
    layer.train(x)
    y = layer.execute(x)
    assert layer.dtype == numx.dtype('f')
    assert y.dtype == layer.dtype


def test_layer_invertibility():
    layer = mh.Layer(_pca_nodes([10, 17, 3], [10, 17, 3]))
    x = numx_rand.random([100, 30]).astype('f')
    layer.train(x)
    y = layer.execute(x)
    x_inverse = layer.inverse(y)
    assert numx.all(numx.absolute(x - x_inverse) < 0.001)


def test_layer_invertibility2():
    # reduce the dimensions, so input_dim != output_dim
    layer = mh.Layer(_pca_nodes([10, 17, 3], [8, 12, 3]))
    x = numx_rand.random([100, 30]).astype('f')
    layer.train(x)
    y = layer.execute(x)
    layer.inverse(y)


def test__same_input_layer():
    layer = mh.SameInputLayer(_pca_nodes([10, 10, 10], [5, 3, 1]))
    x = numx_rand.random([100, 10]).astype('f')
    layer.train(x)
    y = layer.execute(x)
    assert layer.dtype == numx.dtype('f')
    assert y.dtype == layer.dtype


def test_clonelayer():
    node = mdp.nodes.PCANode(input_dim=10, output_dim=5)
    x = numx_rand.random([10, 70]).astype('f')
    layer = mh.CloneLayer(node, 7)
    layer.train(x)
    y = layer.execute(x)
    assert layer.dtype == numx.dtype('f')
    assert y.dtype == layer.dtype


def test_switchboard_inverse1():
    sboard = mh.Switchboard(input_dim=3,
                            connections=[2, 0, 1])
    assert sboard.is_invertible()
    y = numx.array([[2, 3, 4], [5, 6, 7]])
    x = sboard.inverse(y)
    assert numx.all(x == numx.array([[3, 4, 2], [6, 7, 5]]))


def test_switchboard_inverse2():
    sboard = mh.Switchboard(input_dim=3,
                            connections=[2, 1, 1])
    assert not sboard.is_invertible()


# Tests for MeanInverseSwitchboard

def test_mean_inverse_switchboard1():
    sboard = mh.MeanInverseSwitchboard(input_dim=3,
                                       connections=[0, 0, 2])
    assert sboard.is_invertible()
    y = numx.array([[2, 4, 3], [1, 1, 7]])
    x = sboard.inverse(y)
    assert numx.all(x == numx.array([[3, 0, 3], [1, 0, 7]]))


def test_mean_inverse_switchboard2():
    sboard = mh.MeanInverseSwitchboard(input_dim=3,
                                       connections=[1, 1, 1, 2, 2])
    assert sboard.is_invertible()
    y = numx.array([[2, 4, 0, 1, 1], [3, 3, 3, 2, 4]])
    x = sboard.inverse(y)
    assert numx.all(x == numx.array([[0, 2, 1], [0, 3, 3]]))


# Tests for ChannelSwitchboard

def test_out_channel_input():
    sboard = mh.ChannelSwitchboard(input_dim=6,
                                   connections=[5, 5,
                                                0, 1],
                                   out_channel_dim=2,
                                   in_channel_dim=2)
    assert numx.all(sboard.get_out_channel_input(0) ==
                    numx.array([5, 5]))
    assert numx.all(sboard.get_out_channel_input(1) ==
                    numx.array([0, 1]))


def test_out_channels_input_channels():
    sboard = mh.ChannelSwitchboard(input_dim=6,
                                   connections=[5, 5,  # out chan 1
                                                0, 1],  # out chan 2
                                   out_channel_dim=2,
                                   in_channel_dim=2)
    # note that there are 3 input channels
    assert numx.all(sboard.get_out_channels_input_channels(0) ==
                    numx.array([2]))
    assert numx.all(sboard.get_out_channels_input_channels(1) ==
                    numx.array([0]))
    assert numx.all(sboard.get_out_channels_input_channels([0, 1]) ==
                    numx.array([0, 2]))


# Tests for Rectangular2dSwitchboard

def test_rect_2d_routing1():
    sboard = mh.Rectangular2dSwitchboard(in_channels_xy=(3, 2),
                                         in_channel_dim=2,
                                         field_channels_xy=(2, 1),
                                         field_spacing_xy=1)
    assert numx.all(sboard.connections ==
                    numx.array([0, 1, 2, 3, 2, 3, 4, 5, 6, 7,
                                8, 9, 8, 9, 10, 11]))
    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)
    # test generated switchboard
    channel_sboard = sboard.get_out_channel_node(0)
    channel_sboard.execute(x)


def test_rect_2d_routing2():
    sboard = mh.Rectangular2dSwitchboard(in_channels_xy=(2, 4),
                                         in_channel_dim=1,
                                         field_channels_xy=(1, 2),
                                         field_spacing_xy=(1, 2))
    assert numx.all(sboard.connections ==
                    numx.array([0, 2, 1, 3, 4, 6, 5, 7]))
    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)
    # test generated switchboard
    channel_sboard = sboard.get_out_channel_node(0)
    channel_sboard.execute(x)


def test_rect_2d_routing3():
    sboard = mh.Rectangular2dSwitchboard(in_channels_xy=(2, 4),
                                         in_channel_dim=1,
                                         field_channels_xy=2,
                                         field_spacing_xy=(1, 2))
    assert numx.all(sboard.connections == numx.array([0, 1, 2, 3, 4, 5, 6, 7]))


def test_rect_2d_routing4():
    sboard = mh.Rectangular2dSwitchboard(in_channels_xy=4,
                                         in_channel_dim=1,
                                         field_channels_xy=(3, 2),
                                         field_spacing_xy=(1, 2))
    assert numx.all(sboard.connections == numx.array([0, 1, 2, 4, 5, 6,
                                                      1, 2, 3, 5, 6, 7,
                                                      8, 9, 10, 12, 13, 14,
                                                      9, 10, 11, 13, 14, 15]))


def test_rect_2d_get_out_channel_node():
    sboard = mh.Rectangular2dSwitchboard(in_channels_xy=(5, 4),
                                         in_channel_dim=2,
                                         field_channels_xy=(3, 2),
                                         field_spacing_xy=(1, 2))
    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    y = sboard.execute(x)
    # routing layer
    nodes = [sboard.get_out_channel_node(index)
             for index in range(sboard.output_channels)]
    layer = mh.SameInputLayer(nodes)
    layer_y = layer.execute(x)
    assert numx.all((y == layer_y))


def test_rect2d_exception_1():
    bad_args = dict(in_channels_xy=(12, 8),
                    # 3 is the problematic value:
                    field_channels_xy=(4, 3),
                    field_spacing_xy=2,
                    in_channel_dim=3,
                    ignore_cover=False)
    with pytest.raises(mh.Rectangular2dSwitchboardException):
        mh.Rectangular2dSwitchboard(**bad_args)


def test_rect2d_exception_2():
    bad_args = dict(in_channels_xy=(12, 8),
                    # 9 is the problematic value:
                    field_channels_xy=(4, 9),
                    field_spacing_xy=2,
                    in_channel_dim=3,
                    ignore_cover=False)
    with pytest.raises(mh.Rectangular2dSwitchboardException):
        mh.Rectangular2dSwitchboard(**bad_args)


def test_rect2d_exception_3():
    bad_args = dict(in_channels_xy=(12, 8),
                    # 9 is the problematic value:
                    field_channels_xy=(4, 9),
                    field_spacing_xy=2,
                    in_channel_dim=3,
                    ignore_cover=True)
    with pytest.raises(mh.Rectangular2dSwitchboardException):
        mh.Rectangular2dSwitchboard(**bad_args)


# Tests for DoubleRect2dSwitchboard

def test_rect_double_routing_1():
    sboard = mh.DoubleRect2dSwitchboard(in_channels_xy=4,
                                        field_channels_xy=2,
                                        in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                                                       5, 6, 9, 10])))  # uneven fields

    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_rect_double_routing_2():
    sboard = mh.DoubleRect2dSwitchboard(in_channels_xy=(6, 4),
                                        field_channels_xy=(2, 2),
                                        in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, 12, 13, 18, 19,
                                                       14, 15, 20, 21, 16, 17, 22, 23,
                                                       7, 8, 13, 14, 9, 10, 15, 16])))  # uneven fields

    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_rect_double_routing_3():
    sboard = mh.DoubleRect2dSwitchboard(in_channels_xy=(4, 6),
                                        field_channels_xy=2,
                                        in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                                                       16, 17, 20, 21, 18, 19, 22, 23,
                                                       5, 6, 9, 10, 13, 14, 17, 18])))  # uneven fields
    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


# Tests for DoubleRhomb2dSwitchboard

def test_doublerhomb_routing_1():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=(3, 2),
                                         diag_field_channels=2,
                                         in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([1, 6, 7, 4])))
    x = numx.array([list(range(0, sboard.input_dim)), list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_doublerhomd_routing_2():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=(2, 3),
                                         diag_field_channels=2,
                                         in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([6, 2, 3, 7])))
    x = numx.array([list(range(0, sboard.input_dim)), list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_doublerhomd_routing_3():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=(4, 2),
                                         diag_field_channels=2,
                                         in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([1, 8, 9, 5, 2, 9, 10, 6])))
    x = numx.array([list(range(0, sboard.input_dim)), list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_doublerhomd_routing_4():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=(2, 4),
                                         diag_field_channels=2,
                                         in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([8, 2, 3, 9, 9, 4, 5, 10])))
    x = numx.array([list(range(0, sboard.input_dim)), list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_doublerhomd_routing_5():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=4,
                                         diag_field_channels=2,
                                         in_channel_dim=1)
    assert numx.all((sboard.connections == numx.array([1, 16, 17, 5,
                                                       2, 17, 18, 6,
                                                       5, 19, 20, 9,
                                                       6, 20, 21, 10,
                                                       9, 22, 23, 13,
                                                       10, 23, 24, 14])))
    x = numx.array([list(range(0, sboard.input_dim)), list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_doublerhomd_routing_6():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=(7, 4),
                                         diag_field_channels=4,
                                         in_channel_dim=1)
    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_doublerhomd_routing_7():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=(4, 7),
                                         diag_field_channels=4,
                                         in_channel_dim=1)
    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_doublerhomd_routing_8():
    sboard = mh.DoubleRhomb2dSwitchboard(long_in_channels_xy=(6, 7),
                                         diag_field_channels=4,
                                         in_channel_dim=1)
    x = numx.array([list(range(0, sboard.input_dim)),
                    list(range(101, 101 + sboard.input_dim))])
    sboard.execute(x)


def test_hinet_simple_net():
    switchboard = mh.Rectangular2dSwitchboard(in_channels_xy=(12, 8),
                                              field_channels_xy=4,
                                              field_spacing_xy=2,
                                              in_channel_dim=3)

    node = mdp.nodes.PCANode(input_dim=4 * 4 * 3, output_dim=5)
    flownode = mh.FlowNode(mdp.Flow([node, ]))
    layer = mh.CloneLayer(flownode, switchboard.output_channels)
    flow = mdp.Flow([switchboard, layer])
    x = numx_rand.random([5, switchboard.input_dim])
    flow.train(x)

@pytest.fixture
def noisenode():
    return mdp.nodes.NoiseNode(input_dim=20 * 20, noise_args=(0, 0.0001))


def test_sfa_net(noisenode):
    sfa_node = mdp.nodes.SFANode(input_dim=20 * 20, output_dim=10, dtype='f')
    switchboard = mh.Rectangular2dSwitchboard(in_channels_xy=100,
                                              field_channels_xy=20,
                                              field_spacing_xy=10)
    flownode = mh.FlowNode(mdp.Flow([noisenode, sfa_node]))
    sfa_layer = mh.CloneLayer(flownode, switchboard.output_channels)
    flow = mdp.Flow([switchboard, sfa_layer])
    train_gen = numx.cast['f'](numx_rand.random((3, 10, 100 * 100)))
    flow.train([None, train_gen])


def test_hinet_html(noisenode):
    # create some flow for testing
    sfa_node = mdp.nodes.SFANode(input_dim=20 * 20, output_dim=10)
    switchboard = mh.Rectangular2dSwitchboard(in_channels_xy=100,
                                              field_channels_xy=20,
                                              field_spacing_xy=10)
    flownode = mh.FlowNode(mdp.Flow([noisenode, sfa_node]))
    sfa_layer = mh.CloneLayer(flownode, switchboard.output_channels)
    flow = mdp.Flow([switchboard, sfa_layer])
    # create dummy file to write the HTML representation
    html_file = io.StringIO()
    hinet_html = mdp.hinet.HiNetHTMLVisitor(html_file)
    hinet_html.convert_flow(flow)
    html_file.close()


def test_hinetx_html():
    # create some flow for testing
    sfa_node = mdp.nodes.SFANode(input_dim=20 * 20, output_dim=10)
    flow = mdp.Flow([sfa_node])
    # create dummy file to write the HTML representation
    html_file = io.StringIO()
    hinet_html = mdp.hinet.HiNetXHTMLVisitor(html_file)
    hinet_html.convert_flow(flow)
    html_file.close()


def test_layer_training_state():
    # trained sfa nodes
    sfa_nodes1 = [mdp.nodes.SFANode(), mdp.nodes.SFANode(), mdp.nodes.SFANode()]
    [node(numx.random.randn(10, 3)) for node in sfa_nodes1]
    assert (numx.all([not node.is_training() for node in sfa_nodes1]))

    # 2 trained and 1 untrained sfa nodes
    sfa_nodes2 = [mdp.nodes.SFANode(), mdp.nodes.SFANode(input_dim=3), mdp.nodes.SFANode()]
    sfa_nodes2[0](numx.random.randn(10, 3))
    sfa_nodes2[2](numx.random.randn(10, 3))
    assert (not numx.all([not node.is_training() for node in sfa_nodes2]))

    for layer_class in [mdp.hinet.Layer, mdp.hinet.SameInputLayer]:
        # non-trainable layer
        layer_node = layer_class(sfa_nodes1)
        assert (not layer_node.is_training())
        assert (not layer_node.is_trainable())

        # trainable layer
        layer_node = layer_class(sfa_nodes2)
        assert (layer_node.is_training())
        assert (layer_node.is_trainable())

    # non-trainable layer
    layer_node = mdp.hinet.CloneLayer(sfa_nodes1[0], n_nodes=3)
    assert (not layer_node.is_training())
    assert (not layer_node.is_trainable())

    # trainable layer
    layer_node = mdp.hinet.CloneLayer(sfa_nodes2[1])
    assert (layer_node.is_training())
    assert (layer_node.is_trainable())
