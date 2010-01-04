"""
The BiNet package is an extension of the pure feed-forward flow concept in MDP.

It defines a framework for far more general flow sequences, involving 
top-down processes (e.g. for error backpropagation) or even loops.
So the 'bi' in BiNet primarily stands for 'bidirectional'.

BiNet is implemented by extending both the Node and the Flow concept. Both the
new BiNode and BiFlow classes are downward compatible with the classical
Nodes and Flows, allowing them to be combined with BiNet elements. 

The fundamental addition in BiNet is that BiNodes can specify a target node for
their output and that they can send messages to other nodes. A BiFlow is then
needed to interpret these arguments, e.g. to continue the flow execution at the
specified target node.

BiNet is fully integrated with the HiNet and the Parallel packages.


New BiNet concepts: Jumps and Messages
======================================

Jump targets are numbers (relative position in the flow) or strings, which are
then compared to the optional node_id. The target number 0 refers to the node
itself.

Messages are standard Python dictionaries to transport information
that would not fit well into the standard x array. The dict keys also support
target specifications and other magic for more convenient usage.

"""

### T O D O ###

# TODO: add target BiNode unittest
# TODO: add unittests for message parsing, especially magic method key

# TODO: use a special wrapper for classifier nodes

# TODO: provide ParallelBiNode to copy the stop_result attribute?
#    Or can we guarantee that stop_training is always called on the original
#    version? If we relly on this then it should be specified in the API. 

# TODO: nodes should be in binet.nodes?
#    Use subpackages or import everything into binet?

# ------------- optional ----------------

# TODO: implement switchlayer, a layer where each column represents a different
#    target, so the target value determines which nodes are used

# TODO: show more information in trace slides via mouse hover,
#    or enable some kind of folding (might be possible via CSS like suckerfish)

# TODO: use special class for binet results instead of tuples???
# PB: YES! this would get tid of all the if len(results)==5: ...
#    maybe using __slots__?
#    named tuple (2.6) is supposed to be very efficient, I think it uses
#    __slots__
# NW: The situation has imo been mostly settled by the latest simplifications
#    (removing branches). The results are now so simple that I don't think
#    a special result class would be worth it.

# TODO: make comments conform to RST format


from binode import BiNodeException, BiNode, NODE_ID_KEY
from biflow import (MessageResultContainer, BiFlowException, BiFlow,
                    BiCheckpointFlow, EXIT_TARGET)
from binodes import *
from bihinet import *
from inspection import *
from parallel import *
import test

del binode
del binodes
del biflow
del bihinet
del inspection
del parallel
