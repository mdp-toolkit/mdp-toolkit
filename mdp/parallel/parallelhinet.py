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
        """Reference the needed part of the _flow and fork the training node.

        If the fork() of the current node fails the exception is not caught
        here (but will for example be caught in an encapsulating ParallelFlow).
        """
        i_train_node = 0  # index of current training node
        while not self._flow[i_train_node].is_training():
            i_train_node += 1
        node_list = self._flow[:i_train_node]
        node_list.append(self._flow[i_train_node].fork())
        return self.__class__(mdp.Flow(node_list))
    
        i_train_node = 0  # index of current training node
        while not self._flow[i_train_node].is_training():
            i_train_node += 1
            if i_train_node >= len(self._flow):
                i_train_node = -1  # no node in training
                break
        node_list = []
        for i_node, node in enumerate(self._flow):
            if i_node == i_train_node or node.use_fork_execute:
                node_list.append(node.fork())
            else:
                node_list.append(node)
        return self.__class__(self._flow.__class__(node_list))
    

    def _join(self, forked_node):
        """Join the last node from the given forked _flow into this FlowNode."""
        i_train_node = 0  # index of current training node
        while not self._flow[i_train_node].is_training():
            i_train_node += 1
        for i_node, node in  enumerate(forked_node._flow):
            if i_node == i_train_node or node.use_fork_execute:
                self._flow[i_node].join(node)
                
    def purge_nodes(self):
        """Replace nodes that are not forked with None.

        While a purged flow cannot be used to process data any more, it can
        still be joined and can thus save memory and bandwidth.
        """
        for i_node, node in enumerate(self._flow):
            if not (node._train_phase_started or node.use_fork_execute):
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


class ParallelCloneLayer(hinet.CloneLayer, parallelnodes.ParallelExtensionNode):
    """Parallel version of CloneLayer class."""

    def _fork(self):
        """Fork the internal node in the clone layer."""
        return self.__class__(self.node.fork(), n_nodes=len(self.nodes))

    def _join(self, forked_node):
        """Join the internal node in the clone layer."""
        self.node.join(forked_node.node)
