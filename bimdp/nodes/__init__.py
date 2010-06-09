
try:
    from autogen_binodes import *
    from autogen_biclassifiers import *
    del autogen_binodes
    del autogen_biclassifiers
except:
    print "No autogen modules found."
    
from miscnodes import IdentityBiNode, SenderBiNode
from codecorator import CoroutineBiNodeMixin, binode_coroutine

del miscnodes
del codecorator




    
