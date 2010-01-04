
from miscnodes import IdentityBiNode, JumpBiNode, SenderBiNode
from updownnodes import UpDownBiNode, TopUpDownBiNode

del miscnodes
del updownnodes


## automatically create BiNode versions of all Nodes in mdp.nodes ##

# TODO: use a special wrapper for classifier nodes

import sys
import mdp
from ..binode import BiNode

# use a function to avoid poluting the namespace
def _create_binodes():
    current_module = sys.modules[__name__]
    node_metaclass = mdp.NodeMetaclass
    for node_class in (getattr(mdp.nodes, name) for name in dir(mdp.nodes)):
        if not issubclass(type(node_class), node_metaclass):
            continue
        node_name = node_class.__name__
        binode_name = "Bi" + node_name
        docstring = "Automatically created BiNode version of %s." % node_name
        binode_class = node_metaclass.__new__(node_metaclass, binode_name,
                                              (BiNode, node_class),
                                              {"__doc__": docstring})
        setattr(current_module, binode_name, binode_class)
        
_create_binodes()
