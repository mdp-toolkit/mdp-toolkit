"""
Parallel version of bihinet.
"""

import mdp

from bimdp import BiNode, BiNodeException
from bimdp import BiFlow
from bimdp.hinet import BiFlowNode, CloneBiLayer

from parallelbinode import ParallelExtensionBiNode


class BiLearningPhaseNotParallelException(BiNodeException):
    """Exception for unsupported when is_bi_leaning is True."""
    pass


class DummyNode(mdp.Node):
    """Dummy node class for empty nodes."""
     
    def is_trainable(self):
        return False


class ParallelBiFlowNode(BiFlowNode, ParallelExtensionBiNode):
    
    def _fork(self):
        """Fork the training nodes and assemble it with the rest of the flow.
        
        If the fork() of the current node fails the exception is not caught 
        here (but will for example be caught in an encapsulating ParallelFlow). 
        """
        i_train_node = 0  # index of current training node
        while not self._flow[i_train_node].is_training():
            i_train_node += 1
            if i_train_node >= len(self._flow):
                i_train_node = -1  # no node in training
                break
        node_list = []
        for i_node, node in enumerate(self._flow):
            if i_node == i_train_node:
                node_list.append(node.fork())
            elif isinstance(node, BiNode) and node.is_bi_training():
                node_list.append(node.fork())
            else:
                node_list.append(node)
        return ParallelBiFlowNode(BiFlow(node_list))
    
    def _join(self, forked_node):
        """Join the forked node and all bi_forked nodes.
        
        This also works for purged ParallelBiFlowNodes, nodes entries which are 
        None are ignored. 
        """
        i_train_node = 0  # index of current training node
        while not self._flow[i_train_node].is_training():
            i_train_node += 1
        for i_node, node in  enumerate(forked_node._flow):
            if i_node == i_train_node:
                self._flow[i_node].join(node)
            elif isinstance(node, BiNode) and node.is_bi_training():
                self._flow[i_node].join(node)
    
    def purge_nodes(self):
        """Replace nodes that are not training or bi-learning with None.
        
        While a purged flow cannot be used to process data any more, it can 
        still be joined and can thus save memory and bandwidth.
        """
        for i_node, node in enumerate(self._flow):
            if not ((node._train_phase_started) or 
                    (isinstance(node, BiNode) and 
                     node.is_bi_training())):
                self._flow[i_node] = DummyNode()
                
                
class ParallelCloneBiLayer(CloneBiLayer, ParallelExtensionBiNode):
    """Parallel version of CloneBiLayer.
    
    This class also adds support for calling switch_to_instance during training,
    using the join method of the internal nodes.
    """
    
    def _set_use_copies(self, use_copies):
        """Switch internally between using a single node instance or copies.
        
        In a normal CloneLayer a single node instance is used to represent all 
        the horizontally aligned nodes. But in a BiMDP where the nodes store 
        temporary data this may not work. 
        Via this method one can therefore create copies of the single node 
        instance.
        
        This method can also be triggered by the use_copies msg key.
        """
        if use_copies and not self.use_copies:
            # switch to node copies
            self.nodes = [self.node.copy() for _ in range(len(self.nodes))]
            self.node = None  # disable single node while copies are used
            self._uses_copies = True
        elif not use_copies and self.use_copies:
            # switch to a single node instance
            if self.is_training():
                for forked_node in self.nodes[1:]:
                    self.nodes[0].join(forked_node)
            elif self.is_bi_training():
                for forked_node in self.nodes[1:]:
                    self.nodes[0].bi_join(forked_node)
            self.node = self.nodes[0]
            self.nodes = (self.node,) * len(self.nodes) 
    
    def _fork(self):
        """Fork the nodes in the layer to fork the layer."""
        forked_node = ParallelCloneBiLayer( 
                                node=self.nodes[0].fork(), 
                                n_nodes=len(self.nodes), 
                                use_copies=False, 
                                node_id=self._node_id, 
                                dtype=self.get_dtype())
        if self.use_copies:
            # simulate switch_to_copies
            forked_node.nodes = [node.fork() for node in self.nodes]
            forked_node.node = None
            return forked_node
        else:
            return forked_node
        
    def _join(self, forked_node):
        """Join the trained nodes from the forked layer."""
        if self.use_copies:
            for i_node, layer_node in enumerate(self.nodes):
                layer_node.join(forked_node.nodes[i_node])
        else:
            self.node.join(forked_node.node)
