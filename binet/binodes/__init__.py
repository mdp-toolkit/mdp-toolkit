
from miscnodes import IdentityBiNode, JumpBiNode, SenderBiNode
from updownnodes import UpDownBiNode, TopUpDownBiNode

del miscnodes
del updownnodes


## automatically create BiNode versions of all Nodes in mdp.nodes ##

# TODO: use a special wrapper for classifier nodes

# Note: Using mdp.NodeMetaclass.__new__ instead of exec makes the classes
#    appear as if they were defined in mdp.nodes, breaking pickle. 

import sys
import mdp
from ..binode import BiNode

# use a function to avoid poluting the namespace
def _create_binodes():
    current_module = sys.modules[__name__]
    for node_class in (getattr(mdp.nodes, name) for name in dir(mdp.nodes)):
        if (not isinstance(node_class, type) or
            not issubclass(node_class, mdp.Node)):
            continue
        node_name = node_class.__name__
        binode_name = "Bi" + node_name
        docstring = "Automatically created BiNode version of %s." % node_name
        exec ('class %s(BiNode, mdp.nodes.%s): "%s"' %
              (binode_name, node_name, docstring)) in current_module.__dict__
        
_create_binodes()
