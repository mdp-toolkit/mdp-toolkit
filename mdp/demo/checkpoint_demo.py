 

"""This demo shows how to use and define checkpoint functions for
CheckpointFlow.
"""
import mdp

# It can sometimes be useful to execute arbitrary functions at the
# end of the training or execution phase, for example to save the
# internal structures of a node for later analysis. This can easily
# be done defining a ``CheckpointFlow``. As an example imagine the
# following situation: you want to perform Principal Component
# Analysis (PCA) on your data to reduce the dimensionality. After
# this you want to expand the signals into a nonlinear space and then
# perform Slow Feature Analysis to extract slowly varying signals. As
# the expansion will increase the number of components, you don't
# want to run out of memory, but at the same time you want to keep as
# much information as possible after the dimensionality
# reduction. You could do that by specifying the percentage of the
# total input variance that has to be conserved in the dimensionality
# reduction. As the number of output components of the PCA node now
# can become as large as the that of the input components, you want
# to check, after training the PCA node, that this number is below a
# certain threshold. If this is not the case you want to abort the
# execution and maybe start again requesting less variance to be
# kept.
#
# Let start defining a generator to be used through the whole example
# see generators_demo.py for more details about generators.
def gen_data(blocks,dims):
    mat = mdp.numx_rand.random((dims,dims))-0.5
    for i in xrange(blocks):
        # put variables on columns and observations on rows
        block = mdp.utils.mult(mdp.numx_rand.random((1000,dims)), mat)
        yield block
    return

# Define a ``PCANode`` which reduces dimensionality of the input,
# a ``PolynomialExpansionNode`` to expand the signals in the space
# of polynomials of degree 2 and a ``SFANode`` to perform SFA:
#
pca = mdp.nodes.PCANode(output_dim=0.9)
exp = mdp.nodes.PolynomialExpansionNode(2)
sfa = mdp.nodes.SFANode()
# 
# As you see we have set the output dimension of the ``PCANode`` to be ``0.9``.
# This means that we want to keep at least 90% of the variance of the original signal.
# We define a ``PCADimensionExceededException`` that has to be thrown when
# the number of output components exceeds a certain threshold:
class PCADimensionExceededException(Exception):
    """Exception base class for PCA exceeded dimensions case."""
    pass

#
# Then, write a ``CheckpointFunction`` that checks the number of output
# dimensions of the ``PCANode`` and aborts if this number is larger than ``max_dim``:
class CheckPCA(mdp.CheckpointFunction):
    def __init__(self,max_dim):
        self.max_dim = max_dim
    def __call__(self,node):
        act_dim = node.get_output_dim()
        if act_dim > self.max_dim:
            errstr = 'PCA output dimensions exceeded maximum '+\
                     '(%d > %d)'%(act_dim,self.max_dim)
            raise PCADimensionExceededException, errstr
        else:
            print 'PCA output dimensions = %d'%(act_dim)

#
# Define the CheckpointFlow:
flow = mdp.CheckpointFlow([pca, exp, sfa])
# To train it we have to supply 3 generators and 3 checkpoint functions:
try:
    flow.train([gen_data(10, 50), None, gen_data(10, 50)],
               [CheckPCA(10), None, None])
except PCADimensionExceededException,e:
    print 'Expected checkpoint Exception got!\n', str(e)

# The training fails with a ``PCADimensionExceededException``.
# If we only had 12 input dimensions instead of 50 we would have passed
# the checkpoint:
flow[0] = mdp.nodes.PCANode(output_dim=0.9)
flow.train([gen_data(10, 12), None, gen_data(10, 12)],
           [CheckPCA(10), None, None])

# We could use the built-in ``CheckpoinSaveFunction`` to save the ``SFANode`` 
# and analyze the results later :
pca = mdp.nodes.PCANode(output_dim=0.9)
exp = mdp.nodes.PolynomialExpansionNode(2)
sfa = mdp.nodes.SFANode()
flow = mdp.CheckpointFlow([pca, exp, sfa])
flow.train([gen_data(10, 12), None, gen_data(10, 12)],
           [CheckPCA(10),
            None, 
            mdp.CheckpointSaveFunction('dummy.pic',
                                       protocol = 0)])

#
# We can now reload and analyze the ``SFANode``:
fl = file('dummy.pic')
import cPickle
sfa_reloaded = cPickle.load(fl)
print repr(sfa_reloaded)
# Don't forget to clean the rubbish:
fl.close()
import os
os.remove('dummy.pic')


