import mdp
pcanode1 = mdp.nodes.PCANode()
pcanode1
# PCANode(input_dim=None, output_dim=None, dtype=None)
pcanode2 = mdp.nodes.PCANode(output_dim = 10)
pcanode2
# PCANode(input_dim=None, output_dim=10, dtype=None)
pcanode3 = mdp.nodes.PCANode(output_dim = 0.8)
pcanode3.desired_variance
# 0.80000000000000004
pcanode4 = mdp.nodes.PCANode(dtype = 'float32')
pcanode4
# PCANode(input_dim=None, output_dim=None, dtype='float32')
pcanode4.supported_dtypes
# [dtype('float32'), dtype('float64')]
expnode = mdp.nodes.PolynomialExpansionNode(3)
x = mdp.numx_rand.random((100, 25))  # 25 variables, 100 observations
pcanode1.train(x)
pcanode1
# PCANode(input_dim=25, output_dim=None, dtype='float64')
for i in range(100):
    x = mdp.numx_rand.random((100, 25))
    pcanode1.train(x)
# >>>
expnode.is_trainable()
# False
pcanode1.stop_training()
pcanode3.train(x)
pcanode3.stop_training()
pcanode3.output_dim
# 16
pcanode3.explained_variance
# 0.85261144755506446
avg = pcanode1.avg            # mean of the input data
v = pcanode1.get_projmatrix() # projection matrix
fdanode = mdp.nodes.FDANode()
for label in ['a', 'b', 'c']:
    x = mdp.numx_rand.random((100, 25))
    fdanode.train(x, label)
# >>>
fdanode.stop_training()
for label in ['a', 'b', 'c']:
    x = mdp.numx_rand.random((100, 25))
    fdanode.train(x, label)
# >>>
x = mdp.numx_rand.random((100, 25))
y_pca = pcanode1.execute(x)
y_pca = pcanode1(x)
x = mdp.numx_rand.random((100, 5))
y_exp = expnode(x)
x = mdp.numx_rand.random((100, 25))
y_fda = fdanode(x)
pcanode1.is_invertible()
# True
x = pcanode1.inverse(y_pca)
expnode.is_invertible()
# False
class TimesTwoNode(mdp.Node):
    def is_trainable(self):
        return False
    def _execute(self, x):
        return 2*x
    def _inverse(self, y):
        return y/2
# ...
# >>>
node = TimesTwoNode(dtype = 'int32')
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node(x)
print x, '* 2 =  ', y
# [ [ 1.  2.  3.]] * 2 =   [ [2 4 6]]
print y, '/ 2 =', node.inverse(y)
# [ [2 4 6]] / 2 = [ [1 2 3]]
class PowerNode(mdp.Node):
    def __init__(self, power, input_dim=None, dtype=None):
        super(PowerNode, self).__init__(input_dim=input_dim, dtype=dtype)
        self.power = power
    def is_trainable(self):
        return False
    def is_invertible(self):
        return False
    def _get_supported_dtypes(self):
        return ['float32', 'float64']
    def _execute(self, x):
        return self._refcast(x**self.power)
# ...
# >>>
node = PowerNode(3)
x = mdp.numx.array([[1.0, 2.0, 3.0]])
y = node(x)
print x, '**', node.power, '=', node(x)
# [ [ 1.  2.  3.]] ** 3 = [ [  1.   8.  27.]]
class MeanFreeNode(mdp.Node):
    def __init__(self, input_dim=None, dtype=None):
        super(MeanFreeNode, self).__init__(input_dim=input_dim,
                                           dtype=dtype)
        self.avg = None
        self.tlen = 0
    def _train(self, x):
        # Initialize the mean vector with the right
        # size and dtype if necessary:
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.input_dim,
                                      dtype=self.dtype)
        self.avg += mdp.numx.sum(x, axis=0)
        self.tlen += x.shape[0]
    def _stop_training(self):
        self.avg /= self.tlen
        if self.output_dim is None:
            self.output_dim = self.input_dim
    def _execute(self, x):
        return x - self.avg
    def _inverse(self, y):
        return y + self.avg
# ...
# >>>
node = MeanFreeNode()
x = mdp.numx_rand.random((10,4))
node.train(x)
y = node(x)
print 'Mean of y (should be zero): ', mdp.numx.mean(y, 0)
# Mean of y (should be zero):  [  0.00000000e+00   2.22044605e-17
# -2.22044605e-17   1.11022302e-17]
class UnitVarianceNode(mdp.Node):
    def __init__(self, input_dim=None, dtype=None):
        super(UnitVarianceNode, self).__init__(input_dim=input_dim,
                                               dtype=dtype)
        self.avg = None # average
        self.std = None # standard deviation
        self.tlen = 0
    def _get_train_seq(self):
        return [(self._train_mean, self._stop_mean),
                (self._train_std, self._stop_std)]
    def _train_mean(self, x):
        if self.avg is None:
            self.avg = mdp.numx.zeros(self.input_dim,
                                      dtype=self.dtype)
        self.avg += mdp.numx.sum(x, 0)
        self.tlen += x.shape[0]
    def _stop_mean(self):
        self.avg /= self.tlen
    def _train_std(self, x):
        if self.std is None:
            self.tlen = 0
            self.std = mdp.numx.zeros(self.input_dim,
                                      dtype=self.dtype)
        self.std += mdp.numx.sum((x - self.avg)**2., 0)
        self.tlen += x.shape[0]
    def _stop_std(self):
        # compute the standard deviation
        self.std = mdp.numx.sqrt(self.std/(self.tlen-1))
    def _execute(self, x):
        return (x - self.avg)/self.std
    def _inverse(self, y):
        return y*self.std + self.avg
# >>>
node = UnitVarianceNode()
x = mdp.numx_rand.random((10,4))
# loop over phases
for phase in range(2):
    node.train(x)
    node.stop_training()
# ...
# ...
# execute
y = node(x)
print 'Standard deviation of y (should be one): ', mdp.numx.std(y, axis=0)
# Standard deviation of y (should be one):  [ 1.  1.  1.  1.]
class TwiceNode(mdp.Node):
    def is_trainable(self): return False
    def is_invertible(self): return False
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = 2*n
    def _set_output_dim(self, n):
        raise mdp.NodeException, "Output dim can not be set explicitly!"
    def _execute(self, x):
        return mdp.numx.concatenate((x, x), 1)
# ...
# >>>
node = TwiceNode()
x = mdp.numx.zeros((5,2))
x
# array([[0, 0],
# [0, 0],
# [0, 0],
# [0, 0],
# [0, 0]])
node.execute(x)
# array([[0, 0, 0, 0],
# [0, 0, 0, 0],
# [0, 0, 0, 0],
# [0, 0, 0, 0],
# [0, 0, 0, 0]])
inp = mdp.numx_rand.random((1000, 20))
inp = (inp - mdp.numx.mean(inp, 0))/mdp.numx.std(inp, 0)
inp[:,5:] /= 10.0
x = mdp.utils.mult(inp,mdp.numx_rand.random((20, 20)))
inp_test = mdp.numx_rand.random((1000, 20))
inp_test = (inp_test - mdp.numx.mean(inp_test, 0))/mdp.numx.std(inp_test, 0)
inp_test[:,5:] /= 10.0
x_test = mdp.utils.mult(inp_test, mdp.numx_rand.random((20, 20)))
pca = mdp.nodes.PCANode(output_dim=5)
pca.train(x)
out1 = pca(x)
ica = mdp.nodes.CuBICANode()
ica.train(out1)
out2 = ica(out1)
out1_test = pca(x_test)
out2_test = ica(out1_test)
hitnode = mdp.nodes.HitParadeNode(3)
hitnode.train(out2_test)
maxima, indices = hitnode.get_maxima()
flow = mdp.Flow([mdp.nodes.PCANode(output_dim=5), mdp.nodes.CuBICANode()])
flow = mdp.nodes.PCANode(output_dim=5) + mdp.nodes.CuBICANode()
flow.train(x)
flow.append(mdp.nodes.HitParadeNode(3))
flow += mdp.nodes.HitParadeNode(3)
flow.train(x_test)
maxima, indices = flow[2].get_maxima()
out = flow.execute(x)
cov = mdp.numx.amax(abs(mdp.utils.cov2(inp[:,:5], out)), axis=1)
print cov
# [ 0.98992083  0.99244511  0.99227319  0.99663185  0.9871812 ]
out = flow(x)
rec = flow.inverse(out)
cov = mdp.numx.amax(abs(mdp.utils.cov2(x/mdp.numx.std(x,axis=0),
                                       rec/mdp.numx.std(rec,axis=0))))
print cov
# [ 0.99839606  0.99744461  0.99616208  0.99772863  0.99690947
# 0.99864056  0.99734378  0.98722502  0.98118101  0.99407939
# 0.99683096  0.99756988  0.99664384  0.99723419  0.9985529
# 0.99829763  0.9982712   0.99721741  0.99682906  0.98858858]
for node in flow:
    print repr(node)
# ...
# PCANode(input_dim=20, output_dim=5, dtype='float64')
# CuBICANode(input_dim=5, output_dim=5, dtype='float64')
# HitParadeNode(input_dim=5, output_dim=5, dtype='float64')
# HitParadeNode(input_dim=5, output_dim=5, dtype='float64')
# >>>
len(flow)
# 4
print flow[::2]
# [PCANode, HitParadeNode]
nodetoberemoved = flow.pop(-1)
nodetoberemoved
# HitParadeNode(input_dim=5, output_dim=5, dtype='float64')
len(flow)
# 3
dummyflow = flow[1:].copy()
longflow = flow + dummyflow
len(longflow)
# 4
class BogusExceptNode(mdp.Node):
   def train(self,x):
       self.bogus_attr = 1
       raise Exception, "Bogus Exception"
   def execute(self,x):
       raise Exception, "Bogus Exception"
# ...
flow = mdp.Flow([BogusExceptNode()])
flow.set_crash_recovery(1)
flow.set_crash_recovery('/home/myself/mydumps/MDPdump.pic')
class BogusNode(mdp.Node):
    """This node does nothing."""
    def _train(self, x):
        pass
# ...
class BogusNode2(mdp.Node):
    """This node does nothing. But it's not trainable nor invertible.
    """
    def is_trainable(self): return False
    def is_invertible(self): return False
# ...
# >>>
def gen_data(blocks):
    for i in mdp.utils.progressinfo(xrange(blocks)):
        block_x = mdp.numx.atleast_2d(mdp.numx.arange(2,1001,2))
        block_y = mdp.numx.atleast_2d(mdp.numx.arange(1,1001,2))
        # put variables on columns and observations on rows
        block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
        yield block
# ...
# >>>
flow = mdp.Flow([BogusNode(),BogusNode()], verbose=1)
flow.train([gen_data(5000),gen_data(3000)])
# Training node #0 (BogusNode)
# [===================================100%==================================>]
flow = BogusNode() + BogusNode()
block_x = mdp.numx.atleast_2d(mdp.numx.arange(2,1001,2))
block_y = mdp.numx.atleast_2d(mdp.numx.arange(1,1001,2))
single_block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
flow.train(single_block)
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
flow.train([None, gen_data(5000)])
# Training node #0 (BogusNode2)
# Training finished
# Training node #1 (BogusNode)
# [===================================100%==================================>]
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
flow.train(single_block)
# Training node #0 (BogusNode2)
# Training finished
# Training node #1 (BogusNode)
# Training finished
# Close the training phase of the last node
flow = mdp.Flow([BogusNode(),BogusNode()], verbose=1)
flow.train([gen_data(1), gen_data(1)])
# Training node #0 (BogusNode)
# Training finished
# Training node #1 (BosgusNode)
# [===================================100%==================================>]
output = flow(single_block)
output = flow.inverse(single_block)
class SimpleIterable(object):
    def __init__(self, blocks):
        self.blocks = blocks
    def __iter__(self):
        # this is a generator
        for i in range(self.blocks):
            yield generate_some_data()
# >>>
class RandomIterable(object):
    def __init__(self):
        self.state = None
    def __iter__(self):
        if self.state is None:
            self.state = mdp.numx_rand.get_state()
        else:
            mdp.numx_rand.set_state(self.state)
        for i in range(2):
            yield mdp.numx_rand.random((1,4))
iterable = RandomIterable()
for x in iterable:
    print x
# ...
# [[ 0.99586495  0.53463386  0.6306412   0.09679571]]
# [[ 0.51117469  0.46647448  0.95089738  0.94837122]]
for x in iterable:
    print x
# ...
# [[ 0.99586495  0.53463386  0.6306412   0.09679571]]
# [[ 0.51117469  0.46647448  0.95089738  0.94837122]]
def gen_data(blocks,dims):
    mat = mdp.numx_rand.random((dims,dims))-0.5
    for i in xrange(blocks):
        # put variables on columns and observations on rows
        block = mdp.utils.mult(mdp.numx_rand.random((1000,dims)), mat)
        yield block
# ...
# >>>
pca = mdp.nodes.PCANode(output_dim=0.9)
exp = mdp.nodes.PolynomialExpansionNode(2)
sfa = mdp.nodes.SFANode()
class PCADimensionExceededException(Exception):
    """Exception base class for PCA exceeded dimensions case."""
    pass
# ...
# >>>
class CheckPCA(mdp.CheckpointFunction):
    def __init__(self,max_dim):
        self.max_dim = max_dim
    def __call__(self,node):
        node.stop_training()
        act_dim = node.get_output_dim()
        if act_dim > self.max_dim:
            errstr = 'PCA output dimensions exceeded maximum '+\
                     '(%d > %d)'%(act_dim,self.max_dim)
            raise PCADimensionExceededException, errstr
        else:
            print 'PCA output dimensions = %d'%(act_dim)
# ...
# >>>
flow = mdp.CheckpointFlow([pca, exp, sfa])
flow[0] = mdp.nodes.PCANode(output_dim=0.9)
flow.train([gen_data(10, 12), None, gen_data(10, 12)],
           [CheckPCA(10), None, None])
# PCA output dimensions = 6
pca = mdp.nodes.PCANode(output_dim=0.9)
exp = mdp.nodes.PolynomialExpansionNode(2)
sfa = mdp.nodes.SFANode()
flow = mdp.CheckpointFlow([pca, exp, sfa])
flow.train([gen_data(10, 12), None, gen_data(10, 12)],
           [CheckPCA(10),
            None,
            mdp.CheckpointSaveFunction('dummy.pic',
                                       stop_training = 1,
                                       protocol = 0)])
# ...
# PCA output dimensions = 7
fl = file('dummy.pic')
import cPickle
sfa_reloaded = cPickle.load(fl)
sfa_reloaded
# SFANode(input_dim=35, output_dim=35, dtype='d')
fl.close()
import os
os.remove('dummy.pic')
class TestExtensionNode(mdp.ExtensionNode):
    extension_name = "test"
    def _execute(self):
        return 0
# ...
class TestNode(mdp.Node):
    def _execute(self):
        return 1
# ...
class ExtendedTestNode(TestExtensionNode, TestNode):
    pass
# ...
# >>>
node1 = mdp.nodes.PCANode(input_dim=100, output_dim=10)
node2 = mdp.nodes.SFANode(input_dim=100, output_dim=20)
layer = mdp.hinet.Layer([node1, node2])
layer
# Layer(input_dim=200, output_dim=30, dtype=None)
node1_1 = mdp.nodes.PCANode(input_dim=100, output_dim=50)
node1_2 = mdp.nodes.SFANode(input_dim=50, output_dim=10)
node1_flow = mdp.Flow([node1_1, node1_2])
node1 = mdp.hinet.FlowNode(node1_flow)
layer = mdp.hinet.Layer([node1, node2])
layer
# Layer(input_dim=200, output_dim=30, dtype=None)
switchboard = mdp.hinet.Switchboard(input_dim=6, connections=[0,1,2,3,4,3,4,5])
switchboard
# Switchboard(input_dim=3, output_dim=2, dtype=None)
x = mdp.numx.array([[2,4,6,8,10,12]])
switchboard.execute(x)
# array([[ 2,  4,  6,  8, 10,  8, 10, 12]])
mdp.hinet.show_flow(flow)
node1 = mdp.nodes.PCANode(input_dim=100, output_dim=10)
node2 = mdp.nodes.SFA2Node(input_dim=10, output_dim=10)
parallel_flow = mdp.parallel.ParallelFlow([node1, node2])
n_data_chunks = 2
data_iterables = [[mdp.numx_rand.random((200, 100))
                   for _ in range(n_data_chunks)]
                   for _ in range(2)]
scheduler = mdp.parallel.ProcessScheduler(n_processes=2)
parallel_flow.train(data_iterables, scheduler=scheduler)
scheduler.shutdown()
try:
    parallel_flow.train(data_iterables, scheduler=scheduler)
finally:
    scheduler.shutdown()
# ...
p2 = mdp.numx.pi*2
t = mdp.numx.linspace(0,1,10000,endpoint=0) # time axis 1s, samplerate 10KHz
dforce = mdp.numx.sin(p2*5*t) + mdp.numx.sin(p2*11*t) + mdp.numx.sin(p2*13*t)
def logistic_map(x,r):
    return r*x*(1-x)
# ...
# >>>
series = mdp.numx.zeros((10000,1),'d')
series[0] = 0.6
for i in range(1,10000):
    series[i] = logistic_map(series[i-1],3.6+0.13*dforce[i])
# ...
# >>>
flow = (mdp.nodes.EtaComputerNode() +
        mdp.nodes.TimeFramesNode(10) +
        mdp.nodes.PolynomialExpansionNode(3) +
        mdp.nodes.SFANode(output_dim=1) +
        mdp.nodes.EtaComputerNode() )
# ...
# >>>
flow.train(series)
slow = flow(series)
resc_dforce = (dforce - mdp.numx.mean(dforce,0))/mdp.numx.std(dforce,0)
mdp.utils.cov2(resc_dforce[:-9],slow)
# 0.99992501533859179
print 'Eta value (time series): ', flow[0].get_eta(t=10000)
# Eta value (time series):  [ 3002.53380245]
print 'Eta value (slow feature): ', flow[-1].get_eta(t=9996)
# Eta value (slow feature):  [ 10.2185087]
mdp.numx_rand.seed(1266090063)
def uniform(min_, max_, dims):
    """Return a random number between min_ and max_ ."""
    return mdp.numx_rand.random(dims)*(max_-min_)+min_
# ...
def circumference_distr(center, radius, n):
    """Return n random points uniformly distributed on a circumference."""
    phi = uniform(0, 2*mdp.numx.pi, (n,1))
    x = radius*mdp.numx.cos(phi)+center[0]
    y = radius*mdp.numx.sin(phi)+center[1]
    return mdp.numx.concatenate((x,y), axis=1)
# ...
def circle_distr(center, radius, n):
    """Return n random points uniformly distributed on a circle."""
    phi = uniform(0, 2*mdp.numx.pi, (n,1))
    sqrt_r = mdp.numx.sqrt(uniform(0, radius*radius, (n,1)))
    x = sqrt_r*mdp.numx.cos(phi)+center[0]
    y = sqrt_r*mdp.numx.sin(phi)+center[1]
    return mdp.numx.concatenate((x,y), axis=1)
# ...
def rectangle_distr(center, w, h, n):
    """Return n random points uniformly distributed on a rectangle."""
    x = uniform(-w/2., w/2., (n,1))+center[0]
    y = uniform(-h/2., h/2., (n,1))+center[1]
    return mdp.numx.concatenate((x,y), axis=1)
# ...
N = 2000
cf1 = circumference_distr([6,-0.5], 2, N)
cf2 = circumference_distr([3,-2], 0.3, N)
cl1 = circle_distr([-5,3], 0.5, N/2)
cl2 = circle_distr([3.5,2.5], 0.7, N)
r1 = rectangle_distr([-1.5,0], 1, 4, N)
r2 = rectangle_distr([+1.5,0], 1, 4, N)
r3 = rectangle_distr([0,+1.5], 2, 1, N/2)
r4 = rectangle_distr([0,-1.5], 2, 1, N/2)
x = mdp.numx.concatenate([cf1, cf2, cl1, cl2, r1,r2,r3,r4], axis=0)
x = mdp.numx.take(x,mdp.numx_rand.permutation(x.shape[0]), axis=0)
gng = mdp.nodes.GrowingNeuralGasNode(max_nodes=75)
STEP = 500
for i in range(0,x.shape[0],STEP):
    gng.train(x[i:i+STEP])
    # [...] plotting instructions
# ...
gng.stop_training()
n_obj = len(gng.graph.connected_components())
print n_obj
# 5
def s_distr(npoints, hole=False):
    """Return a 3D S-shaped surface. If hole is True, the surface has
    a hole in the middle."""
    t = mdp.numx_rand.random(npoints)
    y = mdp.numx_rand.random(npoints)*5.
    theta = 3.*mdp.numx.pi*(t-0.5)
    x = mdp.numx.sin(theta)
    z = mdp.numx.sign(theta)*(mdp.numx.cos(theta) - 1.)
    if hole:
        indices = mdp.numx.where(((0.3>t) | (0.7<t)) | ((1.>y) | (4.<y)))
        return x[indices], y[indices], z[indices], t[indices]
    else:
        return x, y, z, t
n, k = 1000, 15
x, y, z, t = s_distr(n, hole=False)
data = mdp.numx.array([x,y,z]).T
lle_projected_data = mdp.nodes.LLENode(k, output_dim=2)(data)
x, y, z, t = s_distr(n, hole=True)
data = mdp.numx.array([x,y,z]).T
lle_projected_data = mdp.nodes.LLENode(k, output_dim=2)(data)
hlle_projected_data = mdp.nodes.HLLENode(k, output_dim=2)(data)
import bimdp
pca_node = bimdp.nodes.PCABiNode(node_id="pca")
biflow = bimdp.BiFlow([pca_node])
biflow["pca"]
# PCABiNode(input_dim=None, output_dim=None, dtype=None, node_id="pca")
samples = mdp.numx_rand.random((100,10))
labels = mdp.numx.arange(100)
flow = bimdp.BiFlow([mdp.nodes.PCANode(), bimdp.nodes.FDABiNode()])
flow.train([[samples],[samples]], [None,[{"cl": labels}]])
# git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/mdp-toolkit
# git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/docs
# git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/examples
# git clone git://mdp-toolkit.git.sourceforge.net/gitroot/mdp-toolkit/contrib
