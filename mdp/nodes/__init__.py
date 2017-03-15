# -*- coding:utf-8 -*-
__docformat__ = "restructuredtext en"

from .pca_nodes import WhiteningNode, PCANode
from .sfa_nodes import SFANode, SFA2Node
from .ica_nodes import ICANode, CuBICANode, FastICANode, TDSEPNode
from .neural_gas_nodes import GrowingNeuralGasNode, NeuralGasNode
from .expansion_nodes import (QuadraticExpansionNode, PolynomialExpansionNode,
                             RBFExpansionNode, GrowingNeuralGasExpansionNode,
                             GeneralExpansionNode)
from .fda_nodes import FDANode
from .em_nodes import FANode
from .misc_nodes import (IdentityNode, HitParadeNode, TimeFramesNode,
                        TimeDelayNode, TimeDelaySlidingWindowNode,
                        EtaComputerNode, NoiseNode, NormalNoiseNode,
                        CutoffNode, HistogramNode, AdaptiveCutoffNode, NumxBufferNode,
                         TransformerNode, GridProcessingNode)
from .isfa_nodes import ISFANode
from .rbm_nodes import RBMNode, RBMWithLabelsNode
from .regression_nodes import LinearRegressionNode
from .classifier_nodes import (SignumClassifier, PerceptronClassifier,
                              SimpleMarkovClassifier,
                              DiscreteHopfieldClassifier,
                              KMeansClassifier, GaussianClassifier,
                              NearestMeanClassifier, KNNClassifier)
from .jade import JADENode
from .nipals import NIPALSNode
from .lle_nodes import LLENode, HLLENode
from .xsfa_nodes import XSFANode, NormalizeNode

# import internals for use in test_suites
from .misc_nodes import OneDimensionalHitParade as _OneDimensionalHitParade
from .expansion_nodes import expanded_dim as _expanded_dim

from .mca_nodes_online import MCANode
from .pca_nodes_online import CCIPCANode, CCIPCAWhiteningNode
from .sfa_nodes_online import IncSFANode
from .stats_nodes_online import OnlineCenteringNode, OnlineTimeDiffNode
from .hsfa_nodes import HSFANode

from .basis_function_nodes import BasisFunctionNode

__all__ = ['PCANode', 'WhiteningNode', 'NIPALSNode', 'FastICANode',
           'CuBICANode', 'TDSEPNode', 'JADENode', 'SFANode', 'SFA2Node',
           'ISFANode', 'XSFANode', 'FDANode', 'FANode', 'RBMNode',
           'RBMWithLabelsNode', 'GrowingNeuralGasNode', 'LLENode', 'HLLENode',
           'LinearRegressionNode', 'QuadraticExpansionNode',
           'PolynomialExpansionNode', 'RBFExpansionNode','GeneralExpansionNode',
           'GrowingNeuralGasExpansionNode', 'NeuralGasNode', '_expanded_dim',
           'SignumClassifier',
           'PerceptronClassifier', 'SimpleMarkovClassifier',
           'DiscreteHopfieldClassifier', 'KMeansClassifier', 'NormalizeNode',
           'GaussianClassifier', 'NearestMeanClassifier', 'KNNClassifier',
           'EtaComputerNode', 'HitParadeNode', 'NoiseNode', 'NormalNoiseNode',
           'TimeFramesNode', 'TimeDelayNode', 'TimeDelaySlidingWindowNode',
           'CutoffNode', 'AdaptiveCutoffNode', 'HistogramNode',
           'IdentityNode', '_OneDimensionalHitParade',
           'OnlineCenteringNode', 'OnlineTimeDiffNode', 'CCIPCANode', 'CCIPCAWhiteningNode', 'MCANode',
           'IncSFANode', 'TransformerNode', 'NumxBufferNode', 'BasisFunctionNode', 'GridProcessingNode',
           'HSFANode',
           ]


# nodes with external dependencies
from mdp import config, numx_description, MDPException

if numx_description == 'scipy':
    from .convolution_nodes import Convolution2DNode
    __all__ += ['Convolution2DNode']

if config.has_shogun:
    from .shogun_svm_classifier import ShogunSVMClassifier
    __all__ += ['ShogunSVMClassifier']

if config.has_libsvm:
    from .libsvm_classifier import LibSVMClassifier
    __all__ += ['LibSVMClassifier']

if config.has_sklearn:
    from . import scikits_nodes
    for name in scikits_nodes.DICT_:
        if name.endswith('Node'):
            globals()[name] = scikits_nodes.DICT_[name]
            __all__.append(name)
        del name

if config.has_pyqtgraph:
    from .pg_nodes import PG2DNode, PGCurveNode, PGImageNode
    __all__+= ['PG2DNode', 'PGCurveNode', 'PGImageNode']

if config.has_gym:
    from .openai_gym_nodes import GymNode, GymContinuousExplorerNode
    __all__+=['GymNode', 'GymContinuousExplorerNode']

from mdp import utils
utils.fixup_namespace(__name__, __all__ + ['ICANode'],
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
                       'regression_nodes',
                       'classifier_nodes',
                       'utils',
                       'scikits_nodes',
                       'numx_description',
                       'config',
                       'stats_nodes_online',
                       'pca_nodes_online',
                       'mca_nodes_online',
                       'sfa_nodes_online',
                       'basis_function_nodes',
                       'openai_gym_nodes',
                       'pg_nodes',
                       'hsfa_nodes',
                       ))
