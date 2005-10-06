"""This demo shows how to build, train and execute a flow.

Plots displayed by this demo are completely bogus. They are there only
for illustration purposes.
"""
import sys
import mdp
# A flow consists in an acyclic graph of nodes (currently only
# node sequences are implemented). The data is sent to an 
# input node and is successively processed by the following 
# nodes on the graph. The general flow implementation automatizes 
# the training, execution, and inverse execution (if defined) of 
# the whole graph. Crash recovery is optionally available: in case 
# of failure the current state of the flow is saved for later 
# inspection. A subclass of the basic flow class allows 
# user-supplied checkpoint functions to be executed at the end 
# of each phase, for example to save the internal structures 
# of a node for later analysis.
#
# plot things only if we have scipy and if we are not called by
# the run_all.py script
def plot(x):
    try:
        if __name__ != 'novisual':
            mdp.numx.gplt.plot(x)
    except:
        pass

#
# Flow creation, training and execution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Suppose we have an input signal with an high number of dimensions,
# on which we would like to perform ICA. To make the problem affordable,
# we first need to reduce its dimensionality with PCA. Finally, we would
# like to visualize the data sequence at the beginning and after
# each step.

# We could start by quickly defining a node to visualize the data
# (see node_demo.py for details on subclassing Node)
class VisualizeNode(mdp.Node):
    def is_trainable(self): return 0
    def is_invertible(self): return 0
    def execute(self, x):
        mdp.Node.execute(self,x)
        self._refcast(x)
        plot(x)
        return x
#
# Generate some input signal randomly (which makes the
# example useless, but it's just for illustration...).
# Generate a signal with 20 dimensions and 1000 observations:
inp = mdp.numx_rand.random((1000,20))
# Rescale x to have zero mean and unit variance:
inp = (inp - mdp.utils.mean(inp,0))/mdp.utils.std(inp,0)
# We reduce the variance of the last 15 components, so that they are
# going to be eliminated by PCA:
inp[:,5:] /= 10.0
# Mix linearly  the input signals:
x = mdp.utils.mult(inp,mdp.numx_rand.random((20,20)))
#
# - We could now perform our analysis using only nodes, that's the 
#  lenghty way...
#  1. Visualize the input data:
plot(x)
#  2. Perform PCA:
pca = mdp.nodes.PCANode(output_dim=5)
pca.train(x)
out1 = pca.execute(x)
#  3. Visualize data after PCA:
plot(out1)
#  4. Perform ICA using CuBICA algorithm:
ica = mdp.nodes.CuBICANode()
ica.train(out1)
out2 = ica.execute(out1)
#  5. Visualize data after ICA:
plot(out2)
#
# - ... or we could use flows, the recommended way:
flow = mdp.Flow([VisualizeNode(),
                 mdp.nodes.PCANode(output_dim=5),
                 VisualizeNode(),
                 mdp.nodes.CuBICANode(),
                 VisualizeNode()])
flow.train(x)
out = flow.execute(x)
#   You will probably get some warnings here. This is expected, see the
#   section about Generators_ to learn more about that, for the moment
#   you can simply ignore them.
#
# Just to check that everything works properly, we can 
# calculate covariance between sources and estimated sources
# (should be approximately 1):
cov = mdp.utils.amax(abs(mdp.utils.cov(inp[:,:5],out)))
print 'Covariance between sources and estimated sources:\n',cov
#
# Flow inversion
# ~~~~~~~~~~~~~~
# Flows can be inverted by calling their inverse function.
# In this case, however, the flow contains non-invertible nodes,
# and trying to invert it would raise an exception.
# To overcome this we simply get a slice of the flow instance
# with the invertible nodes.
# Note that a slice of a flow instance returns a new instance
# containing references to the corresponding nodes
# Reconstruct the mix inverting the flow:
rec = flow[1::2].inverse(out)
# Calculate covariance between input mix and reconstructed mix:
# (should be approximately 1)
cov = mdp.utils.amax(abs(mdp.utils.cov(x/mdp.utils.std(x,0),
                                       rec/mdp.utils.std(rec,0))))
print 'Covariance between input mix and reconstructed mix:\n',cov
#
# Flows are container type objects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We have seen that we can get flow slices. Actually flows are Python container
# type objects, very much like lists, i.e. you can loop through them:
for node in flow:
    print repr(node)
#
# You can ``pop``, ``insert`` and ``append`` nodes like you would do with lists:
print len(flow)
nodetoberemoved = flow.pop(-1)
print repr(nodetoberemoved)
print len(flow)
# Finally, you can concatenate flows:
dummyflow = flow[3:].copy()
longflow = flow + dummyflow
print len(longflow)
# The returned flow is always consistent, i.e. input and
# output dimensions of successive nodes always match. If 
# you try to create an inconsistent flow you'll get an
# error:
print repr(flow)
try:
    flow.pop(1)
except ValueError:
    pass

# Crash recovery
# ~~~~~~~~~~~~~~
# If a node in a flow fails, you'll get a traceback that tells you which
# node has failed. You can also switch the crash recovery capability on. If
# something goes wrong you'll end up with a pickle dump of the flow,
# that can be later inspected.
#
# To see how it works let's define a bogus node that always throws an 
# ``Exception`` and put it into a flow:
#
class BogusExceptNode(mdp.Node):
    def train(self,x):
        self.bogus_attr = 1
        raise Exception, "Bogus Exception"
    
    def execute(self,x):
        raise Exception, "Bogus Exception"

flow = mdp.Flow([BogusExceptNode()])
# Switch on crash recovery:
flow.set_crash_recovery(1)
# Attempt to train the flow:
try:
    flow.train([[None]])
except Exception, e:
    if hasattr(e, 'filename'):
        import os
        os.remove(e.filename)

# You can give a file name to tell the flow where to put the dump:
flow.set_crash_recovery('/home/myself/mydumps/MDPdump.pic')
