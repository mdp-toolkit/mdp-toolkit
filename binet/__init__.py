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

### N E X T ###

# TODO: update TopDownNode, use slots for up and down targets,
#    the direction is stored like the canned messages etc.

# TODO: add target BiNode unittest
# TODO: add unittests for message parsing, especially magic method key
 
# TODO: add read-only flow property to access _flow in FlowNode and such?

# TODO: fix unecessary bi_reset calls?
#    Note that bi_reset in BiFlowNode did not work properly until recently.
#    However, it might be better to leave both pre- and post bi_reset in place
#    so that no unecessary data stored (especially when the flow gets pickled).

# TODO: automatically create BiNode versions of all MDP nodes,
#    use exec to define new classes and create fitting docstring,
#    first check is a bi-version is already present

# TODO: show more information in trace slides via mouse hover

# TODO: Node Extensions 
#    implement gradient and parallel via Node Extensions, define
#    NodeExtension metaclass which registers all the available extensions.
#    It is derived from ABCMeta.
#    
#    ParallelNode ABC then derives from Node and has NodeExtension as
#    metaclass. Classes like ParallelSFANode are derived as before.
#
#    Activate all extensions by default??
#
#    All instances of NodeExtension are tracked in a tree structure in
#    NodeExtension. When NodeExtension.activate(ParallelNode) then all
#    tree nodes below ParallelNode are used for extension. If ParallelNode
#    has no abstract methods then it is used as well (and thus provides a
#    default implementation).
#
#    NodeExtension checks that __init__ is not overwritten when a Node is
#    registered.
#
#    Custom Parallel classes can be registered. If a Node is already parallel
#    (this is checked via isinstance) then the activation does not
#    add ParallelNode to the MRO (and deactivate
#    will not remove it, since it is not even present).
#    Provide class decorator to register classes with an extension.
#
#    Flow's that rely on extensions can check that the extension is activated
#    (or automatically activate them). Do the check in __init__. Do not
#    deactivate afterwards, but this can be done manually.
#    Provide a method decorator to check that all nodes have an extension
#    available?

# TODO: gradient extension:
#    gradient extensions and such work via the magic 'method' argument in msg
#    e.g. if "method"="gradient" is present then execute calls _gradient
#    instead of _execute, so _execute is only the default (same for
#    message and _message or stop_message)
#    If x is None then it is not forwarded to the method.
#    Not sure how to implement this yet...

### T O D O ###

# TODO: make comments conform to RST format

# TODO: add msg key wildcard support later on

# TODO: use extension mechanism for backpropagation? 
#    put BackpropagationBiNode on top that compares the
#    incoming x to the reference value in msg, but this node needs no
#    training phase at all, unless you want to control learning e.g. in terms
#    of the reached error.
#    But this cannot be done in a simple parallel fashion.

# TODO: implement switchlayer, a layer where each column represents a different
#    target, so the target value determines which nodes are used


from binode import BiNodeException, BiNode 
from biflow import (MessageResultContainer, BiFlowException, BiFlow,
                    BiCheckpointFlow, EXIT_TARGET)
from binodes import *
from bihinet import *
from inspection import *
from parallel import *

del binode
del binodes
del biflow
del bihinet
del inspection
del parallel
