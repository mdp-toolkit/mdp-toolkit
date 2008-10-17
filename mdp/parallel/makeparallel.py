"""
Module for transforming ('monkey patching') a normal flow into a parallel flow.

Note that this is not yet supported for hinet nodes, since these would require
recursive introspection.
"""

import parallelflows
import parallelnodes


def get_parallel_node_class(node_class):
    """Return the parallel version of a node class.
    
    If no parallel version is found the return value is None.
    """
    for membername in dir(parallelnodes):
        member = parallelnodes.__dict__[membername]
        # check that the member is a class  and that it is derived
        # from ParallelNode
        if (isinstance(member, type) and 
            parallelnodes.ParallelNode in member.mro()):
            # check if one of the member basis is the node_class
            if node_class in member.__bases__:
                return member
    return None
    
def make_flow_parallel(flow):
    """Return a parallel version of the flow.
    
    If available nodes will be modified to their parallel version.
    The original flow and its nodes will not be changed.
    """
    if isinstance(flow, parallelflows.ParallelFlow):
        return flow
    parallel_flow = parallelflows.ParallelFlow([])
    for node in flow:
        node = node.copy()
        node_class = type(node)
        if not isinstance(node, parallelnodes.ParallelNode):
            p_node_class = get_parallel_node_class(node_class)
            if p_node_class is not None:
                node.__class__ = p_node_class
                # store the old node calls for 
                node.__serialclass = node_class
        parallel_flow.append(node)
    parallel_flow.__serialclass = type(flow)
    return parallel_flow
                
def unmake_flow_parallel(parallel_flow):
    """Return a non parallel version of a flow created with make_flow_parallel.
    
    This can only work if the parallel flow was created with make_flow_parallel,
    otherwise an exception is raised.
    """
    try:
        flow_class = parallel_flow.__serialclass
    except:
        err = "The provided flow was not created by make_flow_parallel." 
        raise Exception(err)
    flow = flow_class([])
    for node in parallel_flow:
        node = node.copy()
        try:
            node.__class__ = node.__serialclass
        except:
            pass
        flow.append(node)
    return flow