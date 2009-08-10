"""
The BiNet package is an extension of the pure feed-forward flow concept in MDP.

It defines a framework for far more general flow sequences, involving 
top-down processes (e.g. for error backpropagation) or even loops.
So the 'bi' in BiNet primarily stands for 'bidirectional'.

BiNet is implemented by extending both the Node and the Flow concept. Both the
new BiNode and BiFlow classes are 'downward' compatible with the classical
Nodes and Flows, so they can be combined with BiNet elements. 

The fundamental addition in BiNet is that BiNodes can specify a target node for
their output and that they can send messages to other nodes. A BiFlow is then
needed to interpret these arguments, e.g. to continue the flow execution at the
specified target node.

BiNet is fully integrated with the HiNet and the Parallel packages. This was
actually one main motivation for creating BiNet, to leverage the modular design
of the other MDP packages.

New BiNet concepts:

Jumps and Messages:
===================

Jump targets are numbers (relative position in the flow) or strings, which are
then compared to the optional node_id. The target number 0 refers to the node
itself.

Messages are standard Python dictionaries that contain additional information
that would not fit well into the standard x array. The dict key also support
some more sophisticated target specifications.

Messaging Branches:
===================

In addition to jumps, there is also a limited branching functionality during
training and execution. The limitation is that only a single branch is allowed
at any time. For example we may execute a BiFlow and when a certain node is
reached we want it to send a message to some other node. Of course you could
just specify the second node as the next target value, but this will throw you
out of the normal flow execution. So instead the node can initiate a messaging
branch, in which the message is send to the other node. After the branch has
terminated the normal flow execution will be resumed.

"""

### T O D O ###

# TODO: add target BiNode unittest
# TODO: add unittests for message parsing, especially magic method key

# TODO: binet likelihood should use extension as well
 
# TODO: fix unecessary bi_reset calls?
#    Note that bi_reset in BiFlowNode did not work properly until recently.
#    However, it might be better to leave both pre- and post bi_reset in place
#    so that no unecessary data stored (especially when the flow gets pickled).
#    This is also useful when a previous execution was aborted due to an
#    exception.

# TODO: automatically create BiNode versions of all MDP nodes,
#    use exec to define new classes and create fitting docstring,
#    first check is a bi-version is already present

# TODO: show more information in trace slides via mouse hover,
#    or enable some kind of folding (might be possible via CSS like suckerfish)

# TODO: use special class for binet results instead of tuples???
#    maybe using __slots__?
# PB: YES! this would get tid of all the if len(results)==5: ...

# TODO: make comments conform to RST format

# TODO: add msg key wildcard support later on

# TODO: implement switchlayer, a layer where each column represents a different
#    target, so the target value determines which nodes are used

# TODO: nodes should be in binet.nodes

from binode import BiNodeException, BiNode 
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
