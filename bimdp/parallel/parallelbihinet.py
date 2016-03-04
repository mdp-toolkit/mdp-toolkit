"""
Parallel version of bihinet.
"""
from builtins import range

import mdp
from bimdp.hinet import CloneBiLayer


class ParallelCloneBiLayer(CloneBiLayer, mdp.parallel.ParallelExtensionNode):
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
            
    def use_execute_fork(self):
        if self.use_copies:
            return any(node.use_execute_fork() for node in self.nodes)
        else:
            return self.node.use_execute_fork()
