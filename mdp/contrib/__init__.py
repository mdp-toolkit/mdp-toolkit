from jade import JADENode
from nipals import NIPALSNode
from lle_nodes import LLENode, HLLENode
from xsfa_nodes import XSFANode

try:
    import shogun.Kernel as sgKernel
    import shogun.Features as sgFeatures
    import shogun.Classifier as sgClassifier
    from svm_nodes import ShogunSVMNode
    del svm_nodes
except ImportError:
    pass

del jade
del nipals
del lle_nodes
del xsfa_nodes

__all__ = ['JADENode', 'NIPALSNode', 'LLENode', 'HLLENode', 'XSFANode']
#try:
#    ShogunSVMNode
#    __all__ += ['ShogunSVMNode']
#except NameError:
#   pass    
