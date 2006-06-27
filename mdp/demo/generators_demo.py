## Automatically adapted for numpy Jun 26, 2006 by 

"""This demo shows how to use generators to train and execute flows.
"""
import mdp

# A generator is a Python iterator introduced in Python 2.2 that returns
# a value after each call and can be used for example in ``for`` loops.
# See http://linuxgazette.net/100/pramode.html for an introduction, and
# http://www.python.org/peps/pep-0255.html for a complete description.
#
# Let us define two bogus node classes to be used as examples of nodes:
class BogusNode(mdp.FiniteNode):
    def _train(self, x): pass
    def _stop_training(self): pass
class BogusNode2(mdp.Node):
    """This node does nothing. but it's not trainable and not invertible.
    """
    def is_trainable(self): return 0
    def is_invertible(self): return 0
#
# This generator generates ``blocks`` input blocks to be used as training set.
# In this example one block is a 2-dimensional time-series. The first variable
# is [2,4,6,....,1000] and the second one [0,1,3,5,...,999].
# All blocks are equal, this of course would not be the case in a real-life
# example.
# 
# In this example we use a ``ProgressBar`` to get progress information.
def gen_data(blocks):
    progressbar = mdp.utils.ProgressBar(0,blocks)
    progressbar.update(0)
    for i in xrange(blocks):
        block_x = mdp.utils.atleast_2d(mdp.numx.arange(2,1001,2))
        block_y = mdp.utils.atleast_2d(mdp.numx.arange(1,1001,2))
        # put variables on columns and observations on rows
        block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
        progressbar.update(i+1)
        yield block
    print '\n'
    return
#
# Let's define a bogus flow consisting of 2 ``BogusNode``:
flow = mdp.Flow([BogusNode(),BogusNode()],verbose=1)
#
# Train the first node with 5000 blocks and the second node with 3000 blocks.
# Note that the only allowed argument to ``train`` is a sequence (list or tuple)
# of generators. In case you don't want or need to use incremental learning and
# want to do a one-shot training, you can use as argument to ``train`` a single
# array of data
#
# **block-mode training**
print "Training block-mode..."
flow.train([gen_data(5000),gen_data(3000)])
print "Done..."
print 50*'-'
# **one-shot training** using one set of data for both nodes
flow = mdp.Flow([BogusNode(),BogusNode()])
block_x = mdp.utils.atleast_2d(mdp.numx.arange(2,1001,2))
block_y = mdp.utils.atleast_2d(mdp.numx.arange(1,1001,2))
single_block = mdp.numx.transpose(mdp.numx.concatenate([block_x,block_y]))
print "Training one-shot..."
flow.train(single_block)
print "Done..."
print 50*'-'
#
# If your flow contains non-trainable nodes, you must specify a ``None``
# generator for the non-trainable nodes.
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
print "Training block-mode (second example)..."
flow.train([None,gen_data(5000)])
print 50*'-'

# If in this case you try the one-shot training you'll get two warnings like
# the following ones:
# ! Node 0 in not trainable
# You probably need a 'None' generator for this node. Continuing anyway.
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
print "Training one-shot (second example)..."
flow.train(single_block)
print 50*'-'
#
# You can get rid of this warning either by doing what the warning asks you,
# namely use the generator syntax and provide a ``None`` generator for the
# non-trainable nodes, or by switching off MDP warnings altogether:
import warnings
warnings.filterwarnings("ignore",'.*',mdp.MDPWarning)
flow = mdp.Flow([BogusNode2(),BogusNode()], verbose=1)
print "Training one-shot (second example)..."
flow.train(single_block)
print 50*'-'
#
# To switch on ``MDPWarnings`` again:
warnings.filterwarnings("always",'.*',mdp.MDPWarning)
#
# Generators can be used also for execution (and inversion):
flow = mdp.Flow([BogusNode(),BogusNode()], verbose=1)
flow.train([gen_data(1), gen_data(1)])
print "Executing block-mode..." 
output = flow.execute(gen_data(1000))
print "Done."
print 50*'-'
print "Inversion block-mode..." 
output = flow.inverse(gen_data(1000))
print "Done."
print 50*'-'
#
# Execution and inversion can be done in one-shot mode also. Note that
# since training is finished you are not going to get a warning
print "Executing one-shot..."
output = flow.execute(single_block)
print "Done..."
print 50*'-'
print "Inversion one-shot..."
output = flow.inverse(single_block)
print "Done."
print 50*'-'


    
