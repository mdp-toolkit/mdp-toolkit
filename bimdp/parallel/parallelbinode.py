
import mdp.parallel as parallel

from ..binode import BiNode


class ParallelExtensionBiNode(BiNode, parallel.ParallelExtensionNode):
    
    def fork(self):
        """Return a new instance of this node for remote training or execution.
        
        Unlike a normal ParallelNode a fork is always allowed, not only
        during training.
        """
        return self._fork()