from jade import JADENode
from nipals import NIPALSNode
from lle_nodes import LLENode, HLLENode
from xsfa_nodes import XSFANode

__all__ = ['JADENode', 'NIPALSNode', 'LLENode', 'HLLENode', 'XSFANode']

try:
    from shogun_svm_classifier import ShogunSVMClassifier
    __all__ += ['ShogunSVMClassifier']
    del shogun_svm_classifier
except ImportError:
    pass

try:
    from libsvm_classifier import LibSVMClassifier
    __all__ += ['LibSVMClassifier']
    del libsvm_classifier
except ImportError:
    pass

del jade
del nipals
del lle_nodes
del xsfa_nodes
