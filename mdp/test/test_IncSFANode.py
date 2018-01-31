from builtins import range
from mdp.nodes import PolynomialExpansionNode, SFANode, IncSFANode
from ._tools import *
import time


def test_incsfa_v2():
    iterval = 30
    t = numx.linspace(0, 4 * numx.pi, 500)
    x = numx.zeros([t.shape[0], 2])
    x[:, 0] = numx.real(numx.sin(t) + numx.power(numx.cos(11 * t), 2))
    x[:, 1] = numx.cos(11 * t)
    expnode = PolynomialExpansionNode(2)
    input_data = expnode(x)

    ##Setup node/trainer
    output_dim = 4
    node = IncSFANode(eps=0.05)

    bsfanode = SFANode(output_dim=output_dim)
    bsfanode(input_data)
    bv = bsfanode.sf

    v = []

    _tcnt = time.time()
    for i in range(iterval * input_data.shape[0]):
        node.train(input_data[i % input_data.shape[0]:i % input_data.shape[0] + 1])
        if (node.get_current_train_iteration() % 100 == 0):
            v.append(node.sf)

    print('\nTotal Time for {} iterations: {}'.format(iterval, time.time() - _tcnt))

    dcosines = numx.zeros([len(v), output_dim])
    for i in range(len(v)):
        for dim in range(output_dim):
            dcosines[i, dim] = numx.fabs(numx.dot(v[i][:, dim], bv[:, dim].T)) / (
                numx.linalg.norm(v[i][:, dim]) * numx.linalg.norm(bv[:, dim]))
    assert_almost_equal(numx.ones(output_dim), dcosines[-1], decimal=2)


def test_incsfanode_numx_rng():
    x = mdp.numx_rand.randn(100, 5)

    numx_rng = mdp.numx_rand.RandomState(seed=10)
    node1 = IncSFANode(numx_rng=numx_rng)
    node1.train(x)
    init_wv1 = node1.init_pca_vectors
    init_mv1 = node1.init_mca_vectors
    init_sf1 = node1.init_slow_features
    v1 = node1.sf

    numx_rng = mdp.numx_rand.RandomState(seed=10)
    node2 = IncSFANode(numx_rng=numx_rng)
    node2.train(x)
    init_wv2 = node2.init_pca_vectors
    init_mv2 = node2.init_mca_vectors
    init_sf2 = node2.init_slow_features
    v2 = node2.sf

    assert_array_equal(init_wv1, init_wv2)
    assert_array_equal(init_mv1, init_mv2)
    assert_array_equal(init_sf1, init_sf2)
    assert_array_equal(v1, v2)
