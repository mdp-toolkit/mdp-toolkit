"""
Module for transforming ('monkey patching') a normal flow into a parallel flow.

Note that constructing a parallel flow directly is the safer and better way.
This module should only be used if the simplicity is important and only
standard MDP nodes are used (including hinet). 
"""

import mdp
import mdp.hinet as hinet

import parallelflows
import parallelhinet
import parallelnodes


def get_parallel_member(nonparallel_class, parallel_module):
    """Return the parallel version of a node class.
    
    If no parallel version is found the return value is None.
    
    nonparallel_class -- Class of the non-parallel object.
    parallel_module -- Module in which the parallel class might be found.
    """
    for membername in dir(parallel_module):
        member = parallel_module.__dict__[membername]
        # check that the member is a class  and that it is derived
        # from ParallelNode
        if (isinstance(member, type) and 
            member.__name__.startswith("Parallel")):
            # check if the parallel node is directly derived from the
            # node_class, directly because otherwise we might accidently pick
            # a specialized subclass (e.g. ParallelCheckpointFlow instead of 
            # ParallelFlow, since both are derived from Flow)
            if nonparallel_class in member.__bases__:
                return member
    return None


class HiNetParallelTranslator(hinet.HiNetTranslator):
    """Translator to turn a normal flow into a parallel one.
    
    This is done recursively for HiNet structures. Normal nodes are translated
    by simply replacing their __class__ with the parallel class version.
    """
    
    def make_flow_parallel(self, flow):
        """Return a parallel version of the provided flow.
    
        If a parallel version of an internal node is found the node is
        transformed into it. This happens in-place, so flow is modified (after
        the translation it is the original Flow instance, but with parallel
        nodes inside).
        """
        return self._translate_flow(flow)
    
    # overwrite private methods
    
    def _translate_flow(self, flow):
        """Return a parallel version of the provided flow."""
        if isinstance(flow, parallelflows.ParallelFlow):
            return flow
        parallel_nodes = super(HiNetParallelTranslator, 
                               self)._translate_flow(flow)
        flow_class = type(flow)
        flow_parallel_class = get_parallel_member(flow_class, parallelflows)
        parallel_flow = flow_parallel_class(parallel_nodes,  
                                            verbose=flow.verbose)
        parallel_flow._serialclass_ = flow_class
        return parallel_flow
    
    def _translate_flownode(self, flownode):
        """Replace a FlowNode with its parallel version."""
        parallel_nodes = super(HiNetParallelTranslator, 
                               self)._translate_flownode(flownode)
        return parallelhinet.ParallelFlowNode(mdp.Flow(parallel_nodes))
    
    def _translate_layer(self, layer):
        """Replace a Layer with its parallel version."""
        parallel_nodes = super(HiNetParallelTranslator, 
                               self)._translate_layer(layer)
        return parallelhinet.ParallelLayer(parallel_nodes)
        
    def _translate_clonelayer(self, clonelayer):
        """Replace a CloneLayer with its parallel version."""
        parallel_nodes = super(HiNetParallelTranslator, 
                               self)._translate_clonelayer(clonelayer)
        return parallelhinet.ParallelCloneLayer(node=parallel_nodes[0],
                                                n_nodes=len(parallel_nodes))

    def _translate_standard_node(self, node):
        """Try to find a corresponding parallel node class and use that."""
        node_class = type(node)
        if not isinstance(node, parallelnodes.ParallelNode):
            p_node_class = get_parallel_member(node_class, parallelnodes)
            if p_node_class is not None:
                node.__class__ = p_node_class
                # store the old node class for a later back-translation 
                node._serialclass_ = node_class
        return node
    

class HiNetUnParallelTranslator(hinet.HiNetTranslator):
    """Inverse translator for parallel flows from HiNetParallelTranslator.
    
    This is done recursively for HiNet structures. Parallel nodes are translated
    by simply replacing their __class__ with the normal class version.
    """
    
    def unmake_flow_parallel(self, flow):
        """Return the original non-parallel version of parallel_flow.
    
        This can only work if the parallel flow was created with
        make_flow_parallel, otherwise an exception is raised.
        The node translation happens in-place, so parallel_flow is modified
        """
        return self._translate_flow(flow)
        
    # overwrite private methods
    
    def _translate_flow(self, flow):
        """Translate flow into the non-parallel original flow."""
        try:
            flow_class = flow._serialclass_
        except:
            err = "The provided flow was not created by make_flow_parallel." 
            raise Exception(err)
        normal_nodes = super(HiNetUnParallelTranslator, 
                             self)._translate_flow(flow)
        return flow_class(normal_nodes, verbose=flow.verbose)
    
    def _translate_flownode(self, flownode):
        """Replace a parallel FlowNode with its normal version."""
        normal_nodes = super(HiNetUnParallelTranslator, 
                             self)._translate_flownode(flownode)
        return hinet.FlowNode(mdp.Flow(normal_nodes))
    
    def _translate_layer(self, layer):
        """Replace a parallel Layer with its normal version."""
        normal_nodes = super(HiNetUnParallelTranslator, 
                             self)._translate_layer(layer)
        return hinet.Layer(normal_nodes)
        
    def _translate_clonelayer(self, clonelayer):
        """Replace a parallel CloneLayer with its normal version."""
        normal_nodes = super(HiNetUnParallelTranslator, 
                             self)._translate_clonelayer(clonelayer)
        return hinet.CloneLayer(node=normal_nodes[0],
                                n_nodes=len(normal_nodes))

    def _translate_standard_node(self, node):
        """Restore the original node class."""
        try:
            node.__class__ = node._serialclass_
        except:
            pass
        return node


# simple helper functions #

def make_flow_parallel(flow):
    """Return a parallel version of the provided flow.
    
    If a parallel version of an internal node is found the node is transformed
    into it. This happens in-place, so flow is modified (after the translation
    it is the original Flow instance, but with parallel nodes inside).
    """
    translator = HiNetParallelTranslator()
    return translator.make_flow_parallel(flow)
                
def unmake_flow_parallel(parallel_flow):
    """Return the original non-parallel version of parallel_flow.
    
    This can only work if the parallel flow was created with make_flow_parallel,
    otherwise an exception is raised.
    The node translation happens in-place, so parallel_flow is modified
    """
    translator = HiNetUnParallelTranslator()
    return translator.unmake_flow_parallel(parallel_flow)