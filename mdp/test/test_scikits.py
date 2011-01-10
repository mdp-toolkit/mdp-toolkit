from _tools import *

requires_scikits = skip_on_condition(
    "not mdp.config.has_scikits",
    "This test requires Scikits")

@requires_scikits
def test_scikits_PCANode_training():
    node = mdp.nodes.PCAScikitsLearnNode(output_dim=3)
    x = numx.concatenate((numx_rand.randn(100, 10),
                          10 + numx_rand.randn(300, 10)))
    node.train(x)
    node.stop_training()
    y = node.execute(x)

    x.shape, y.shape
    #node.prob(x)

#GMMNode = wrap_scikits_alg(scikits.learn.glm.ElasticNet)
#node = GMMNode(alpha=0., dtype='f')
#x = numpy.random.rand(100,5).astype('f')
#y = 0.3 * x[:,1] + 0.7 * x[:,4]
#node.train(x, y)
#node.stop_training()
#z = node.execute(x)
