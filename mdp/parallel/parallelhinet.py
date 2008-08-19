"""
Parallel versions of hinet nodes.
"""

import mdp
import mdp.hinet as hinet

import parallelnodes

# TODO: Better checks or exception handling for non-parallel nodes in layers.

class ParallelFlowNode(hinet.FlowNode, parallelnodes.ParallelNode):
    """Parallel version of FlowNode."""
    
    def _fork(self):
        """Copy the needed part of the _flow and fork the training node.
        
        If the fork() of the current node fails the exception is not caught 
        here (but will for example be caught in an encapsulating ParallelFlow). 
        """
        i_train_node = 0  # index of current training node
        while not self._flow[i_train_node].is_training():
            i_train_node += 1
        if isinstance(self._flow[i_train_node], parallelnodes.ParallelNode):
            node_list = [self._flow[i].copy() for i in range(i_train_node)]
            node_list.append(self._flow[i_train_node].fork())
        else:
            text = ("Non-parallel node no. %d in ParallelFlowNode." % 
                    (i_train_node+1))
            raise parallelnodes.TrainingPhaseNotParallelException(text)
        return ParallelFlowNode(mdp.Flow(node_list))
    
    def _join(self, forked_node):
        """Join the last node from the given forked _flow into this FlowNode."""
        i_node = len(forked_node._flow) - 1
        self._flow[i_node].join(forked_node._flow[i_node])


class ParallelLayer(hinet.Layer, parallelnodes.ParallelNode):
    """Parallel version of a Layer."""
    
    def _fork(self):
        """Fork or copy all the nodes in the layer to fork the layer."""
        forked_nodes = []
        for node in self.nodes:
            if node.is_training():
                forked_nodes.append(node.fork())
            else:
                forked_nodes.append(node.copy())
        return ParallelLayer(forked_nodes)
        
    def _join(self, forked_node):
        """Join the trained nodes from the forked layer."""
        for i_node, layer_node in enumerate(self.nodes):
            if layer_node.is_training():
                layer_node.join(forked_node.nodes[i_node])


class ParallelCloneLayer(hinet.CloneLayer, parallelnodes.ParallelNode):
    """Parallel version of CloneLayer class."""
    
    def _fork(self):
        """Fork the internal node in the clone layer."""
        return ParallelCloneLayer(self.node.fork(), n_nodes=len(self.nodes))
    
    def _join(self, forked_node):
        """Join the internal node in the clone layer."""
        self.node.join(forked_node.node)
        
        