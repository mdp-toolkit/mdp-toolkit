 

"""This demo shows hot to use nodes.
"""
import mdp

# Nodes
# -----
# A node is the basic unit in MDP and it represents a data processing
# element, like for example a learning algorithm, a filter, a
# visualization step etc. Each node can have a training phase, during 
# which the internal structures are learned from training data (e.g. 
# the weights of a neural network are adapted or the covariance matrix
# is estimated) and an execution phase, where new data can be processed
# forwards (by processing the data through the node) or backwards (by 
# applying the inverse of the transformation computed by the node if 
# defined). MDP is designed to make the implementation of new algorithms
# easy and intuitive, for example by setting automatically input and 
# output dimension and by casting the data to match the typecode 
# (e.g. float or double precision) of the internal structures. Most of 
# the nodes were designed to be applied to arbitrarily long sets of 
# data: the internal structures can be updated successively by 
# sending chunks of the input data (this is equivalent to online 
# learning if the chunks consists of single observations, or to 
# batch learning if the whole data is sent in a single chunk). 
# Already implemented nodes include Principal Component Analysis
# (PCA), Independent Component Analysis (ICA), Slow Feature 
# Analysis (SFA), and Growing Neural Gas Network.
#
# Node Creation
# ~~~~~~~~~~~~~
# Nodes can be obtained by creating an instance of the node class.
# Each node is characterized by an input dimension, that corresponds
# to the dimensionality of the input vectors, an output dimension, and
# a typecode, which determines the typecode of the internal structures
# and of the output signal. These three attributes are inherited from
# the input data if left unspecified. Input dimension and typecode
# can usually be specified when an instance of the node class
# is created.
# The constructor of each node class can require other task-specific
# arguments.
#
# Some examples of node creation:
#
# - Create a node that performs Principal Component Analysis (PCA) 
#   whose input dimension and typecode
#   are inherited from the input data during training. Output dimensions
#   default to input dimensions.
pcanode1 = mdp.nodes.PCANode()
print repr(pcanode1)
#
# - Setting ``output_dim = 10`` means that the node will keep only the 
#   first 10 principal components of the input.
pcanode2 = mdp.nodes.PCANode(output_dim = 10)
print repr(pcanode2)
#
# - If the typecode is set to ``f`` (float) the input 
#   data is cast to float precision when received and the internal 
#   structures are also stored as ``f``. The typecode influences the 
#   memory space necessary for a node and the precision with which the 
#   computations are performed.
pcanode3 = mdp.nodes.PCANode(typecode = 'f')
print repr(pcanode3)
#
#   You can obtain a list of the typecodes supported by a node
#   by calling its 'get_supported_typecodes' method:
print pcanode3.get_supported_typecodes()
#
# - A PolynomialExpansionNode expands its input in the space
#   of polynomals of a given degree by computing all monomials up
#   to the specified degree. Its constructor needs as first argument
#   the degree of the polynomials space (3 in this case).
expnode = mdp.nodes.PolynomialExpansionNode(3)
#
# Node Training
# ~~~~~~~~~~~~~
# Some nodes need to be trained to perform their task. This can
# be done during a training phase by calling the ``train`` method.
#
# - Create some random data and update the internal structures
#   (i.e. mean and covariance matrix) of the PCANode:
x = mdp.numx_rand.random((100, 25))  # 25 variables, 100 observations
pcanode1.train(x)
#
#   At this point the input dimension and the typecode have been
#   inherited from ``x``:
print repr(pcanode1)
#
# - We can train our node with more than one chunk of data. This
#   is especially useful when the input data is too long to
#   be stored in memory or when it has to be created on-the-fly.
#   (See also generators_demo.py)
for i in range(100):
    x = mdp.numx_rand.random((100, 25))
    pcanode1.train(x)
#
# - Some nodes don't need to be trained:
print expnode.is_trainable()
#
#   Trying to train them anyway would raise an exception:
x = mdp.numx_rand.random((100, 5))
try:
    expnode.train(x)
except mdp.IsNotTrainableException, e:
    print 'IsNotTrainableException:',  str(e)
#
# - The training phase ends when the ``stop_training``, ``execute``, or
#   ``inverse`` method are called. For example we can stop the training 
#   of the PCANode (at this point the principal components are computed):
pcanode1.stop_training()
#
#   It is now possible to access the trained internal data
avg = pcanode1.avg            # mean of the input data
v = pcanode1.get_projmatrix() # projection matrix
#
# Node Execution
# ~~~~~~~~~~~~~~
# After the training phase it is possible to execute the node:
#
# - The input data is projected on the principal components learned
#   in the training phase.
x = mdp.numx_rand.random((100, 25))
y_pca = pcanode1.execute(x)
# - Calling a node instance is equivalent to executing it:
y_pca = pcanode1(x)
#
# - The input data is expanded in the space of polynomials of
#   degree 3.
x = mdp.numx_rand.random((100, 5))
y_exp = expnode(x)
#
# Node Inversion
# ~~~~~~~~~~~~~~
# If the operation computed by the node is invertible, it is possible
# to compute the inverse transformation:
#
# - Given the output data, compute the inverse projection to
#   the input space for the PCA node:
print pcanode1.is_invertible()
x = pcanode1.inverse(y_pca)
#
# - The expansion node in not invertible:
print expnode.is_invertible()
#
#   Trying to compute the inverse would raise an exception:
try:
    expnode.inverse(y_exp)
except mdp.IsNotInvertibleException, e:
    print 'IsNotInvertibleException:', str(e)

