"""
Module for MDP classifiers that support parallel training.
"""

import mdp
from mdp.parallel import ParallelExtensionNode


class ParallelGaussianClassifier(ParallelExtensionNode,
                                 mdp.nodes.GaussianClassifier):

    def _fork(self):
        return self._default_fork()

    def _join(self, forked_node):
        if not self._cov_objs:
            self.set_dtype(forked_node._dtype)
            self._cov_objs = forked_node._cov_objs
        else:
            for key, forked_cov in forked_node._cov_objs.items():
                if key in self._cov_objs:
                    self._join_covariance(self._cov_objs[key], forked_cov)
                else:
                    self._cov_objs[key] = forked_cov
                    

#class ParallelNearestMeanClassifier(ParallelExtensionNode,
#                                    mdp.nodes.NearestMeanClassifier):

#class ParallelKNNClassifier(ParallelExtensionNode,
#                            mdp.nodes.KNNClassifier):


