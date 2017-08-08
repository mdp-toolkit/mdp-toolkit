from _tools import *
import tempfile
import cPickle
import py.test


def assert_array_almost_equal_upto_sign(x, y, decimal=6):
    try:
        assert_array_almost_equal(x, y, decimal)
    except AssertionError:
        assert_array_almost_equal(-x, y, decimal)


def assert_almost_equal_upto_sign(x, y, decimal=6):
    try:
        assert_almost_equal(x, y, decimal)
    except AssertionError:
        assert_almost_equal(-x, y, decimal)


def test_single_sfa():
    hsfanode = mdp.nodes.HSFANode(in_channels_xy=(3, 3), field_channels_xy=[(-1, -1)], field_spacing_xy=-1,
                                  n_features=5, in_channel_dim=1, n_training_fields=None, noise_sigma=0.0,
                                  output_dim=4)
    sfanode = mdp.nodes.SFANode(output_dim=hsfanode.output_dim)
    x = mdp.numx.random.randn(100, hsfanode.input_dim)
    assert (hsfanode.get_remaining_train_phase() == 1)
    out1 = hsfanode(x)
    out2 = sfanode(x)
    assert_array_almost_equal_upto_sign(out1, out2)


def test_single_hinet_node():
    hsfanode = mdp.nodes.HSFANode(in_channels_xy=(3, 3), field_channels_xy=[(-1, -1)], field_spacing_xy=-1,
                                  n_features=[(5, 5)], in_channel_dim=1, n_training_fields=None, noise_sigma=0.0,
                                  output_dim=4)
    sfanode = mdp.nodes.SFANode(output_dim=5) + mdp.nodes.QuadraticExpansionNode() \
              + mdp.nodes.SFANode(output_dim=hsfanode.output_dim)
    x = mdp.numx.random.randn(100, hsfanode.input_dim)
    out1 = hsfanode(x)
    out2 = sfanode(x)

    assert_array_almost_equal_upto_sign(out1, out2)


def test_full_hsfa_training_with_save_load():
    t = mdp.numx.linspace(0, 4 * mdp.numx.pi, 1000)
    x = mdp.numx.zeros([t.shape[0], 2])
    x[:, 0] = mdp.numx.real(mdp.numx.sin(t) + mdp.numx.power(mdp.numx.cos(11 * t), 2))
    x[:, 1] = mdp.numx.real(mdp.numx.cos(11 * t))
    x = mdp.numx.dot(x, mdp.numx.random.randn(2, 49))

    hsfanode = mdp.nodes.HSFANode(in_channels_xy=(7, 7), field_channels_xy=[(3, 3), (2, 2), (-1, -1)],
                                  field_spacing_xy=[(2, 2), (1, 1), (-1, -1)],
                                  n_features=[(5, 10), (10, 40), (10, 40)], n_training_fields=None)

    for i in xrange(hsfanode.get_remaining_train_phase()):
        hsfanode.train(x)
        hsfanode.stop_training()

    dummy_file = tempfile.mktemp(prefix='MDP_', suffix=".pic", dir=py.test.mdp_tempdirname)
    # dummy_file = tempfile.mktemp(prefix='MDP_', suffix=".pic", dir='/tmp/')
    hsfanode.save(dummy_file)
    f = open(dummy_file, 'r')
    hsfanode_copy = cPickle.load(f)
    out = hsfanode_copy.execute(x)

    # check for the first two features.
    feats = [mdp.numx.real(mdp.numx.sin(t)), mdp.numx.real(mdp.numx.cos(2 * t))]
    for _i in xrange(len(feats)):
        assert_array_almost_equal_upto_sign(out[:, _i] / out[:, _i].max(), feats[_i], decimal=2)


def test_container_methods():
    t = mdp.numx.linspace(0, 4 * mdp.numx.pi, 1000)
    x = mdp.numx.zeros([t.shape[0], 2])
    x[:, 0] = mdp.numx.real(mdp.numx.sin(t) + mdp.numx.power(mdp.numx.cos(11 * t), 2))
    x[:, 1] = mdp.numx.real(mdp.numx.cos(11 * t))
    x = mdp.numx.dot(x, mdp.numx.random.randn(2, 49))

    hsfanode = mdp.nodes.HSFANode(in_channels_xy=(7, 7), field_channels_xy=[(3, 3), (2, 2), (-1, -1)],
                                  field_spacing_xy=[(2, 2), (1, 1), (-1, -1)],
                                  n_features=[(5, 10), (10, 40), (10, 40)], n_training_fields=None)

    for i in xrange(hsfanode.get_remaining_train_phase()):
        hsfanode.train(x)
        hsfanode.stop_training()

    assert (len(hsfanode) == 3)
    assert (hsfanode._train_phase == 6)
    assert (hsfanode.get_remaining_train_phase() == 0)
    assert mdp.numx.all([not node.is_training() for node in hsfanode])
    assert not hsfanode.is_training()

    hsfanode_new = mdp.nodes.HSFANode(in_channels_xy=(7, 7), field_channels_xy=[(3, 3), (2, 2), (-1, -1)],
                                      field_spacing_xy=[(2, 2), (1, 1), (-1, -1)],
                                      n_features=[(6, 10), (10, 40), (10, 30)], n_training_fields=None,
                                      output_dim=30)

    assert (hsfanode_new._train_phase == 0)
    assert (hsfanode_new.get_remaining_train_phase() == 6)

    def _check_fn1(node_copy, node_ref):
        assert (node_copy._train_phase == 2)
        assert (node_copy.get_remaining_train_phase() == 4)
        tr_states = [node.is_training() for node in node_copy]
        assert_array_equal(tr_states, [False, True, True])
        assert node_copy.output_dim == node_ref.output_dim
        assert node_copy.n_features[:-2] != node_ref.n_features[:-2]
        assert node_copy.n_features[-2:] == node_ref.n_features[-2:]
        assert node_copy.is_training()

    # replace the top two layers of the trained hsfa copy with the untrained hsfa
    hsfanode_copy = hsfanode.copy()
    hsfanode_copy[-2:] = hsfanode_new[-2:]
    _check_fn1(hsfanode_copy, hsfanode_new)

    # delete the last two layers of hsfa copy and then append them the untrained hsfa ones
    hsfanode_copy = hsfanode.copy()
    del hsfanode_copy[-2:]
    hsfanode_copy.extend(hsfanode_new[-2:])
    _check_fn1(hsfanode_copy, hsfanode_new)

    # delete the last two layers of hsfa copy and then append them the untrained hsfa ones
    hsfanode_copy = hsfanode.copy()
    del hsfanode_copy[-2:]
    hsfanode_copy.append(hsfanode_new[-2])
    hsfanode_copy.append(hsfanode_new[-1])
    _check_fn1(hsfanode_copy, hsfanode_new)

    # replace the bottom two layers of the untrained hsfa with the trained hsfa
    hsfanode_new[:2] = hsfanode[:2]
    assert (hsfanode_new._train_phase == 4)
    assert (hsfanode_new.get_remaining_train_phase() == 2)
    tr_states = [node.is_training() for node in hsfanode_new]
    assert_array_equal(tr_states, [False, False, True])
    assert hsfanode_new.output_dim != hsfanode.output_dim
    assert hsfanode_new.n_features[:2] == hsfanode.n_features[:2]
    assert hsfanode_new.n_features[2:] != hsfanode.n_features[2:]
    assert hsfanode_new.is_training()


def test_container_methods_same_network_outputs():
    t = mdp.numx.linspace(0, 4 * mdp.numx.pi, 1000)
    x = mdp.numx.zeros([t.shape[0], 2])
    x[:, 0] = mdp.numx.real(mdp.numx.sin(t) + mdp.numx.power(mdp.numx.cos(11 * t), 2))
    x[:, 1] = mdp.numx.real(mdp.numx.cos(11 * t))
    x = mdp.numx.dot(x, mdp.numx.random.randn(2, 49))

    hsfanode = mdp.nodes.HSFANode(in_channels_xy=(7, 7), field_channels_xy=[(3, 3), (2, 2), (-1, -1)],
                                  field_spacing_xy=[(2, 2), (1, 1), (-1, -1)],
                                  n_features=[(5, 10), (10, 40), (10, 40)], n_training_fields=None)

    for i in xrange(hsfanode.get_remaining_train_phase()):
        hsfanode.train(x)
        hsfanode.stop_training()

    hsfanode_new = mdp.nodes.HSFANode(in_channels_xy=(7, 7), field_channels_xy=[(3, 3), (2, 2), (-1, -1)],
                                      field_spacing_xy=[(2, 2), (1, 1), (-1, -1)],
                                      n_features=[(5, 10), (10, 40), (10, 40)], n_training_fields=None)

    # replace the top two layers of the trained hsfa copy with the untrained hsfa
    hsfanode_copy = hsfanode.copy()
    hsfanode_copy[-2:] = hsfanode_new[-2:]

    # finally train the copy
    for _ in xrange(hsfanode_copy.get_remaining_train_phase()):
        hsfanode_copy.train(x)
        hsfanode_copy.stop_training()

    # compare the first output feature
    out1 = hsfanode.execute(x)
    out2 = hsfanode_copy.execute(x)
    assert_almost_equal_upto_sign(out1[:, 0], out2[:, 0], decimal=2)


def test_random_sampled_hsfa():
    t = mdp.numx.linspace(0, 4 * mdp.numx.pi, 1000)
    x = mdp.numx.zeros([t.shape[0], 2])
    x[:, 0] = mdp.numx.real(mdp.numx.sin(t) + mdp.numx.power(mdp.numx.cos(11 * t), 2))
    x[:, 1] = mdp.numx.real(mdp.numx.cos(11 * t))
    x = mdp.numx.dot(x, mdp.numx.random.randn(2, 49))

    hsfanode = mdp.nodes.HSFANode(in_channels_xy=(7, 7), field_channels_xy=[(3, 3), (2, 2), (-1, -1)],
                                  field_spacing_xy=[(2, 2), (1, 1), (-1, -1)],
                                  n_features=[(5, 10), (10, 40), (10, 40)], n_training_fields=-1)

    blksize = 100
    n_blocks = int(x.shape[0] / blksize)
    for _ in xrange(hsfanode.get_remaining_train_phase()):
        for block_i in xrange(n_blocks):
            hsfanode.train(x[block_i * blksize: (block_i + 1) * blksize])
        hsfanode.stop_training()

    for i in xrange(hsfanode.get_remaining_train_phase()):
        hsfanode.train(x)
        hsfanode.stop_training()

    hsfanode_new = mdp.nodes.HSFANode(in_channels_xy=(7, 7), field_channels_xy=[(3, 3), (2, 2), (-1, -1)],
                                      field_spacing_xy=[(2, 2), (1, 1), (-1, -1)],
                                      n_features=[(5, 10), (10, 40), (10, 40)], n_training_fields=None)

    for _ in xrange(hsfanode_new.get_remaining_train_phase()):
        hsfanode_new.train(x)
        hsfanode_new.stop_training()

    # compare the output
    out1 = hsfanode.execute(x)
    out2 = hsfanode_new.execute(x)
    assert_almost_equal_upto_sign(out1[:, 0], out2[:, 0], decimal=2)
