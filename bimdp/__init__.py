"""
The BiMDP package is an extension of the pure feed-forward flow concept in MDP.

It defines a framework for far more general flow sequences, involving
top-down processes (e.g. for error backpropagation) or even loops.
So the 'bi' in BiMDP primarily stands for 'bidirectional'.

BiMDP is implemented by extending both the Node and the Flow concept. Both the
new BiNode and BiFlow classes are downward compatible with the classical
Nodes and Flows, allowing them to be combined with BiMDP elements.

The first fundamental addition in BiMDP is that BiNodes can specify a target
node for their output, to continue the flow execution at the specified target
node. The second new feature is that Nodes can use messages to propagate
arbitrary information, in addition to the standard single array data.
A BiFlow is needed to enable these features, and the BiNode class has adds
convenience functionality to help with this.

Another important addition are the inspection capapbilities (e.g.,
bimdo.show_training), which create and interactive HTML representation of the
data flow. This makes debugging much easier and can also be extended to
visualize data (see the demos in the test folder).

BiMDP fully supports and extends the HiNet and the Parallel packages.


New BiMDP concepts: Jumps and Messages
======================================

Jump targets are numbers (relative position in the flow) or strings, which are
then compared to the optional node_id. The target number 0 refers to the node
itself.
During execution a node can also use the value of EXIT_TARGET (which is
currently just 'exit') as target value to end the execution. The BiFlow
will then return the last output as result.

Messages are standard Python dictionaries to transport information
that would not fit well into the standard x array. The dict keys also support
target specifications and other magic for more convenient usage.
This is described in more detail in the BiNode module.
"""

### T O D O ###

# TODO: add a target seperator that does not remove the key. Could use
#    -> remove key
#    --> remove one '-' on entry
#    => do not remove the key
#  Note that adding this kind of magic is relatively cheap in BiNode,
#  in parsing first check just for > .

# TODO: add wildcard support for node_id in message keys.
#    Simply tread the node_id part of the key as a regex and check for match.
#    This adds an overhead of about 1 sec per 100,000 messages.

# TODO: Terminate execution if both x and msg are None? This could help in
#    the stop_training execution, but could lead to strange results
#    during normal execution.
#    We could add a check before the result is returned in execute.

# TODO: support dictionary methods like 'keys' in BiFlow? 

# TODO: add workaround for Google Chrome issue once a solution for
#    http://code.google.com/p/chromium/issues/detail?id=47416
#    is in place.

# TODO: Implement more internal checks for node output result?
#    Check that last element is not None?

# TODO: implement switchlayer, a layer where each column represents a different
#    target, so the target value determines which nodes are used

# TODO: show more information in trace slides via mouse hover,
#    or enable some kind of folding (might be possible via CSS like suckerfish)


from .binode import (
    BiNodeException, BiNode, PreserveDimBiNode, MSG_ID_SEP, binode_coroutine
)
from .biclassifier import BiClassifier
from .biflow import (
    MessageResultContainer, BiFlowException, BiFlow, BiCheckpointFlow,
    EXIT_TARGET
)
# the inspection stuff is considered a core functionality
from .inspection import *

from .test import test

from . import nodes
from . import hinet
from . import parallel

del binode
del biflow
del inspection

from mdp.utils import fixup_namespace
fixup_namespace(__name__, None,
                ('binode',
                 'biclassifier',
                 'biflow',
                 'inspection',
                 ))
del fixup_namespace
