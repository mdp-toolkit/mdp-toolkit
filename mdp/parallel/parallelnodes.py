"""
Module for MDP Nodes that support parallel training.

Note that such ParallelNodes are only needed for training, parallel execution
works with any Node that can be pickled.

This module contains both the parallel base class and some parallel 
implementations of MDP nodes. 

WARNING: There is a problem with unpickled arrays in NumPy < 1.1.x, see
http://projects.scipy.org/scipy/numpy/ticket/551
To circumvent this, you can use a copy() of all unpickled arrays. 
"""

import mdp


class TrainingPhaseNotParallelException(mdp.NodeException):
    """Exception for parallel nodes that do not support fork() in some phases.
    
    This exception signals that training should be done locally for this
    training phase. Only when this exception, when raised by fork(), is caught
    in ParallelFlow for local training. 
    """
    pass


class ParallelNode(mdp.Node):
    """Base class for parallel trainable MDP nodes."""
    
    def fork(self):
        """Return a (modified) copy of this node for remote training.
        
        The forked node should be a ParallelNode as well, thus allowing
        recursive forking and joining.
        The actual forking is implemented in _fork.
        """
        if not self.is_trainable():
            raise mdp.IsNotTrainableException, "This node is not trainable."
        if not self.is_training():
            raise mdp.TrainingFinishedException, \
                  "The training phase has already finished."
        return self._fork()
    
    # TODO: check that the dimensions match, allow late setting
    def join(self, forked_node):
        """Absorb the trained node from a fork into this parent node.
        
        The actual joining is implemented in _join.
        """
        if not self.is_trainable():
            raise mdp.IsNotTrainableException, "This node is not trainable."
        if not self.is_training():
            raise mdp.TrainingFinishedException, \
                  "The training phase has already finished."
        if self.dtype == None:
            self.dtype = forked_node.dtype
        if self.input_dim == None:
            self.input_dim = forked_node.input_dim
        if self.output_dim == None:
            self.output_dim = forked_node.output_dim
        if not self._train_phase_started:
            self._train_phase_started = True
        self._join(forked_node)
            
    ## hook methods, overwrite these ##
    
    def _fork(self):
        """Hook method for forking, to be overridden."""
        raise TrainingPhaseNotParallelException("fork is not implemented " +
                                                "by this node.")
    
    def _join(self, forked_node):
        """Hook method for joining, to be overridden."""
        # be aware of http://projects.scipy.org/scipy/numpy/ticket/551
        # copy arrays after unpickling for numpy version < 1.1.x
        raise TrainingPhaseNotParallelException("join is not implemented " +
                                                "by this node.")
    

## MDP parallel node implementations ##

    
class ParallelPCANode(mdp.nodes.PCANode, ParallelNode):
    """Parallel version of MDP PCA node."""
    
    # Warning: __init__ and _fork must be updated when the arguments 
    #    of the corresponding PCANode methods change.
    
    def __init__(self, input_dim=None, output_dim=None, dtype=None, svd=False):
        """Initialize the node.
        
        The reduce argument is not supported, since the results may varry for
        different forks.
        """
        mdp.nodes.PCANode.__init__(self, input_dim=input_dim, 
                                   output_dim=output_dim, dtype=dtype,
                                   svd=svd, reduce=False)
    
    def _fork(self):
        """Fork the node and (if necessary) init the covariance matrices."""
        forked_node = ParallelPCANode(input_dim=self.input_dim, 
                                      output_dim=self.output_dim, 
                                      dtype=self.dtype,
                                      svd=self.svd)
        return forked_node
    
    def _join(self, forked_node):
        """Combine the covariance matrices."""
        if self._cov_mtx._cov_mtx == None:
            self.set_dtype(self._cov_mtx._dtype)
            self._cov_mtx = forked_node._cov_mtx
        else:
            self._cov_mtx._cov_mtx += forked_node._cov_mtx._cov_mtx
            self._cov_mtx._avg += forked_node._cov_mtx._avg
            self._cov_mtx._tlen += forked_node._cov_mtx._tlen
            
            
class ParallelWhiteningNode(mdp.nodes.WhiteningNode, ParallelPCANode):
    
    # Warning: __init__ and _fork must be updated when the arguments 
    #    of the corresponding WhiteningNode methods change.
    
    def __init__(self, input_dim=None, output_dim=None, dtype=None, svd=False):
        """Initialize the node.
        
        The reduce argument is not supported, since the results may varry for
        different forks.
        """
        ParallelPCANode.__init__(self, input_dim=input_dim, 
                                 output_dim=output_dim, dtype=dtype, svd=svd)
    
    def _fork(self):
        """Fork the node and (if necessary) init the covariance matrices."""
        forked_node = ParallelWhiteningNode(input_dim=self.input_dim, 
                                            output_dim=self.output_dim, 
                                            dtype=self.dtype,
                                            svd=self.svd)
        return forked_node
    
    
class ParallelSFANode(mdp.nodes.SFANode, ParallelNode):
    """Parallel version of MDP SFA node."""
    
    def _fork(self):
        """Fork the node and (if necessary) init the covariance matrices."""
        forked_node = ParallelSFANode(input_dim=self.input_dim, 
                                      output_dim=self.output_dim, 
                                      dtype=self.dtype)
        return forked_node
    
    def _join(self, forked_node):
        """Combine the covariance matrices."""
        if self._cov_mtx._cov_mtx == None:
            self.set_dtype(self._cov_mtx._dtype)
            self._cov_mtx = forked_node._cov_mtx
            self._dcov_mtx = forked_node._dcov_mtx
        else:
            self._cov_mtx._cov_mtx += forked_node._cov_mtx._cov_mtx
            self._cov_mtx._avg += forked_node._cov_mtx._avg
            self._cov_mtx._tlen += forked_node._cov_mtx._tlen
            self._dcov_mtx._cov_mtx += forked_node._dcov_mtx._cov_mtx
            self._dcov_mtx._avg += forked_node._dcov_mtx._avg
            self._dcov_mtx._tlen += forked_node._dcov_mtx._tlen
            
            
class ParallelSFA2Node(mdp.nodes.SFA2Node, ParallelSFANode):
    """Parallel version of MDP SFA2Node."""
    
    def _fork(self):
        """Fork the node and (if necessary) init the covariance matrices."""
        forked_node = ParallelSFA2Node(input_dim=self.input_dim, 
                                       output_dim=self.output_dim, 
                                       dtype=self.dtype)
        return forked_node
            
            
