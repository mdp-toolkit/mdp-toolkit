from pca_nodes import WhiteningNode, PCANode
from sfa_nodes import SFANode, SFA2Node
from ica_nodes import ICANode, CuBICANode, FastICANode, TDSEPNode
from neural_gas_nodes import GrowingNeuralGasNode
from expansion_nodes import (QuadraticExpansionNode, PolynomialExpansionNode,
                             RBFExpansionNode, GrowingNeuralGasExpansionNode)
from fda_nodes import FDANode
from em_nodes import FANode
from misc_nodes import (IdentityNode, HitParadeNode, TimeFramesNode,
                        EtaComputerNode, NoiseNode, NormalNoiseNode,
                        GaussianClassifierNode,
                        CutoffNode, HistogramNode, AdaptiveCutoffNode)
from isfa_nodes import ISFANode
from rbm_nodes import RBMNode, RBMWithLabelsNode
from regression_nodes import LinearRegressionNode
from classifier_nodes import (SignumClassifier, PerceptronClassifier,
                              SimpleMarkovClassifier,
                              DiscreteHopfieldClassifier,
                              KMeansClassifier)
from jade import JADENode
from nipals import NIPALSNode
from lle_nodes import LLENode, HLLENode
from xsfa_nodes import XSFANode

# import internals for use in test_suites
from misc_nodes import OneDimensionalHitParade as _OneDimensionalHitParade
from expansion_nodes import expanded_dim as _expanded_dim

__all__ = ['CuBICANode', 'EtaComputerNode', 'FANode', 'FDANode', 'FastICANode',
           'GaussianClassifierNode', 'GrowingNeuralGasNode', 'HitParadeNode',
           'ICANode', 'ISFANode', 'NoiseNode', 'NormalNoiseNode',
           'IdentityNode',
           'PCANode', 'PolynomialExpansionNode', 'QuadraticExpansionNode',
           'RBFExpansionNode', 'RBMNode', 'RBMWithLabelsNode', 'SFA2Node',
           'SFANode', 'TDSEPNode', 'TimeFramesNode','WhiteningNode',
           'LinearRegressionNode', '_OneDimensionalHitParade', '_expanded_dim',
           'CutoffNode', 'HistogramNode', 'AdaptiveCutoffNode',
           'SignumClassifier', 'PerceptronClassifier',
           'SimpleMarkovClassifier', 'DiscreteHopfieldClassifier',
           'KMeansClassifier', 'JADENode', 'NIPALSNode', 'LLENode', 'HLLENode',
           'XSFANode']

# nodes with external dependencies
from mdp import config

if config.module_exists('scipy.signal'):
    from convolution_nodes import Convolution2DNode
    __all__ += ['Convolution2DNode']
    del convolution_nodes

if config.has('shogun'):
    from shogun_svm_classifier import ShogunSVMClassifier
    __all__ += ['ShogunSVMClassifier']
    del shogun_svm_classifier
    del svm_classifiers

if config.has('LibSVM'):
    from libsvm_classifier import LibSVMClassifier
    __all__ += ['LibSVMClassifier']
    del libsvm_classifier
    del svm_classifiers

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
del regression_nodes
del classifier_nodes
del jade
del nipals
del lle_nodes
del xsfa_nodes
del config

from mdp import utils
import sys
del utils
del sys
