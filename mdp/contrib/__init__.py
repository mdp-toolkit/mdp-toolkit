from jade import JADENode
from nipals import NIPALSNode
from lle_nodes import LLENode, HLLENode
from xsfa_nodes import XSFANode

__all__ = ['JADENode', 'NIPALSNode', 'LLENode', 'HLLENode', 'XSFANode']

try:
    from shogun_svm_node import ShogunSVMNode
    __all__ += ['ShogunSVMNode']
    del shogun_svm_node
except ImportError:
    pass

try:
    from libsvm_node import LibSVMNode
    __all__ += ['LibSVMNode']
    del libsvm_node
except ImportError:
    pass

del jade
del nipals
del lle_nodes
del xsfa_nodes
