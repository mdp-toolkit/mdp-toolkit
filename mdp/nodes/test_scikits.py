import mdp
import numpy

node = mdp.nodes.scikits.PCANode(output_dim=3)
x = numpy.concatenate((numpy.random.randn(100, 10), 10 + numpy.random.randn(300, 10)))
node.train(x)
node.stop_training()
y = node.execute(x)
print x.shape, y.shape
#print node.prob(x)

#GMMNode = wrap_scikits_alg(scikits.learn.glm.ElasticNet)
#node = GMMNode(alpha=0., dtype='f')
#x = numpy.random.rand(100,5).astype('f')
#y = 0.3 * x[:,1] + 0.7 * x[:,4]
#node.train(x, y)
#node.stop_training()
#z = node.execute(x)
