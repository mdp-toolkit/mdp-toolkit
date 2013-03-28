"""MDP extension to support masked node data.
"""
__docformat__ = "restructuredtext en"

import mdp

class MaskedCovarianceNode(mdp.ExtensionNode):
    """MDP extension for masked training data

    Adapt a node to handle masked training data.
    """
    extension_name = 'masked'

    def _new_covariance_matrix(self, *args, **kwargs):
        return mdp.utils.MaskedCovarianceMatrix(dtype=self.dtype)

class MaskedPCANode(MaskedCovarianceNode, mdp.nodes.PCANode):
    pass

class MaskedGaussianClassifier(MaskedCovarianceNode,
                               mdp.nodes.GaussianClassifier):
    pass

class MaskedFANode(MaskedCovarianceNode, mdp.nodes.FANode):
    pass

class MaskedFDANode(MaskedCovarianceNode, mdp.nodes.FDANode):
    pass

class MaskedSFANode(MaskedCovarianceNode, mdp.nodes.SFANode):
    pass

class MaskedXSFANode(MaskedCovarianceNode, mdp.nodes.XSFANode):
    pass


