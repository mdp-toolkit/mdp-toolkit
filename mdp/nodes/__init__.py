__docformat__ = "restructuredtext en"

from pca_nodes import WhiteningNode, PCANode
from sfa_nodes import SFANode, SFA2Node
from ica_nodes import ICANode, CuBICANode, FastICANode, TDSEPNode
from neural_gas_nodes import GrowingNeuralGasNode
from expansion_nodes import (QuadraticExpansionNode, PolynomialExpansionNode,
                             RBFExpansionNode, GrowingNeuralGasExpansionNode,
                             GeneralExpansionNode)
from fda_nodes import FDANode
from em_nodes import FANode
from misc_nodes import (IdentityNode, HitParadeNode, TimeFramesNode,
                        TimeDelayNode, TimeDelaySlidingWindowNode,
                        EtaComputerNode, NoiseNode, NormalNoiseNode,
                        CutoffNode, HistogramNode, AdaptiveCutoffNode)
from isfa_nodes import ISFANode
from rbm_nodes import RBMNode, RBMWithLabelsNode
from regression_nodes import LinearRegressionNode
from classifier_nodes import (SignumClassifier, PerceptronClassifier,
                              SimpleMarkovClassifier,
                              DiscreteHopfieldClassifier,
                              KMeansClassifier, GaussianClassifier,
                              NearestMeanClassifier, KNNClassifier)
from jade import JADENode
from nipals import NIPALSNode
from lle_nodes import LLENode, HLLENode
from xsfa_nodes import XSFANode

# import internals for use in test_suites
from misc_nodes import OneDimensionalHitParade as _OneDimensionalHitParade
from expansion_nodes import expanded_dim as _expanded_dim

__all__ = ['PCANode', 'WhiteningNode', 'NIPALSNode', 'FastICANode',
           'CuBICANode', 'TDSEPNode', 'JADENode', 'SFANode', 'SFA2Node',
           'ISFANode', 'XSFANode', 'FDANode', 'FANode', 'RBMNode',
           'RBMWithLabelsNode', 'GrowingNeuralGasNode', 'LLENode', 'HLLENode',
           'LinearRegressionNode', 'QuadraticExpansionNode',
           'PolynomialExpansionNode', 'RBFExpansionNode','GeneralExpansionNode',
           'GrowingNeuralGasExpansionNode', '_expanded_dim', 'SignumClassifier',
           'PerceptronClassifier', 'SimpleMarkovClassifier',
           'DiscreteHopfieldClassifier', 'KMeansClassifier',
           'GaussianClassifier', 'NearestMeanClassifier', 'KNNClassifier',
           'EtaComputerNode', 'HitParadeNode', 'NoiseNode', 'NormalNoiseNode',
           'TimeFramesNode', 'TimeDelayNode', 'TimeDelaySlidingWindowNode',
           'CutoffNode', 'AdaptiveCutoffNode', 'HistogramNode',
           'IdentityNode', '_OneDimensionalHitParade']

# nodes with external dependencies
from mdp import config, numx_description, MDPException

if numx_description == 'scipy':
    from convolution_nodes import Convolution2DNode
    __all__ += ['Convolution2DNode']
    del convolution_nodes

if config.has_shogun:
    from shogun_svm_classifier import ShogunSVMClassifier
    __all__ += ['ShogunSVMClassifier']
    del shogun_svm_classifier

if config.has_libsvm:
    from libsvm_classifier import LibSVMClassifier
    __all__ += ['LibSVMClassifier']
    del libsvm_classifier

if config.has_scikits:
    import scikits_nodes
    for name in scikits_nodes.DICT_:
        if name.endswith('Node'):
            globals()[name] = scikits_nodes.DICT_[name]
            __all__.append(name)
    try:
        del name
    except:
        pass
    del scikits_nodes
try:
    del svm_classifiers
except NameError:
    pass


# clean up namespace
del expansion_nodes
del pca_nodes
del sfa_nodes
del ica_nodes
del neural_gas_nodes
del fda_nodes
del em_nodes
del misc_nodes
del isfa_nodes
del rbm_nodes
del nipals
del numx_description
del config
del MDPException
del jade
del lle_nodes
del xsfa_nodes
del regression_nodes
del classifier_nodes

from mdp import utils
utils.fixup_namespace(__name__, __all__,
                      ('pca_nodes',
                       'sfa_nodes',
                       'ica_nodes',
                       'neural_gas_nodes',
                       'expansion_nodes',
                       'fda_nodes',
                       'em_nodes',
                       'misc_nodes',
                       'isfa_nodes',
                       'rbm_nodes',
                       'regression_nodes',
                       'classifier_nodes',
                       'jade',
                       'nipals',
                       'lle_nodes',
                       'xsfa_nodes',
                       'convolution_nodes',
                       'shogun_svm_classifier',
                       'svm_classifiers',
                       'libsvm_classifier',
                       'svn_classifiers',
                       'regression_nodes',
                       'classifier_nodes',
                       ))
del utils
