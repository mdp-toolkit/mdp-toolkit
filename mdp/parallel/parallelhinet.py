"""
Parallel versions of hinet nodes.

Note that internal nodes are referenced instead of copied, in order to save
memory.
"""

import mdp
import mdp.hinet as hinet

import parallelnodes


class DummyNode(mdp.Node):
    """Dummy node class for empty nodes."""

    @staticmethod
    def is_trainable():
        return False


class ParallelFlowNode(hinet.FlowNode, parallelnodes.ParallelExtensionNode):
    """Parallel version of FlowNode."""

    def _fork(self):
        """Fork nodes that require it, reference all other nodes.

        If a required fork() fails the exception is not caught here.
        """   
        node_list = []
        for node in self._flow:
            if node.is_training() or node.use_execute_fork:
                node_list.append(node.fork())
            else:
                node_list.append(node)
        return self.__class__(self._flow.__class__(node_list))
    
    def _join(self, forked_node):
        """Join the last node from the given forked _flow into this FlowNode."""             
        for i_node, node in enumerate(forked_node._flow):
            if node.is_training() or node.use_execute_fork:
                self._flow[i_node].join(node)
                
    def use_execute_fork(self):
        return any(node.use_execute_fork for node in self._flow)
    
    # TODO: move this to a custom class in parallelflows        
    def purge_nodes(self):
        """Replace nodes that are not forked with None.

        While a purged flow cannot be used to process data any more, it can
        still be joined and can thus save memory and bandwidth.
        """
        for i_node, node in enumerate(self._flow):
            if not (node._train_phase_started or node.use_execute_fork):
                self._flow[i_node] = DummyNode()


class ParallelLayer(hinet.Layer, parallelnodes.ParallelExtensionNode):
    """Parallel version of a Layer."""

    def _fork(self):
        """Fork or copy all the nodes in the layer to fork the layer."""
        forked_nodes = []
        for node in self.nodes:
            if node.is_training():
                forked_nodes.append(node.fork())
            else:
                forked_nodes.append(node)
        return self.__class__(forked_nodes)

    def _join(self, forked_node):
        """Join the trained nodes from the forked layer."""
        for i_node, layer_node in enumerate(self.nodes):
            if layer_node.is_training():
                layer_node.join(forked_node.nodes[i_node])
                
    def use_execute_fork(self):
        return any(node.use_execute_fork for node in self.nodes)


class ParallelCloneLayer(hinet.CloneLayer, parallelnodes.ParallelExtensionNode):
    """Parallel version of CloneLayer class."""

    def _fork(self):
        """Fork the internal node in the clone layer."""
        return self.__class__(self.node.fork(), n_nodes=len(self.nodes))

    def _join(self, forked_node):
        """Join the internal node in the clone layer."""
        self.node.join(forked_node.node)
        
    def use_execute_fork(self):
        return self.node.use_execute_fork
