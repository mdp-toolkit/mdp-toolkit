"""
Module for MDP Nodes that support parallel training.

This module contains both the parallel base class and some parallel 
implementations of MDP nodes. Note that such ParallelNodes are only needed for 
training, parallel execution works with any Node that can be pickled.
"""

# WARNING: There is a problem with unpickled arrays in NumPy < 1.1.x, see
# http://projects.scipy.org/scipy/numpy/ticket/551
# To circumvent this, you can use a copy() of all unpickled arrays. 

import mdp
from mdp import numx

import cPickle as pickle


class TrainingPhaseNotParallelException(mdp.NodeException):
    """Exception for parallel nodes that do not support fork() in some phases.
    
    This exception signals that training should be done locally for this
    training phase. Only when this exception, when raised by fork(), is caught
    in ParallelFlow for local training. 
    """
    pass

class JoinParallelNodeException(mdp.NodeException):
    """Exception for errors when joining parallel nodes."""
    pass


class ParallelNode(mdp.Node):
    """Base class for parallel trainable MDP nodes.
    
    With the fork method new node instances are created which can then be
    trained. With the join method the trained instances are then merged back
    into a single node instance.
    
    Since fork typically has to create a new class instance it must know all
    __init__ arguments. Therefore the __init__ method of a parallel node should
    not accept **kwargs. Otherwise some newly added arguments might be
    ignored in forked nodes, resulting in errors that are very hard to track
    down.
    """
    
    def fork(self):
        """Return a new instance of this node class for remote training.
        
        This is a template method, the actual forking should be implemented in
        _fork.
        
        The forked node should be a ParallelNode of the same class as well, 
        thus allowing recursive forking and joining.
        """
        if not self.is_trainable():
            raise mdp.IsNotTrainableException, "This node is not trainable."
        if not self.is_training():
            raise mdp.TrainingFinishedException, \
                  "The training phase has already finished."
        return self._fork()
    
    # TODO: check that the dimensions match?
    def join(self, forked_node):
        """Absorb the trained node from a fork into this parent node.
        
        This is a template method, the actual joining should be implemented in
        _join.
        """
        if not self.is_trainable():
            raise mdp.IsNotTrainableException, "This node is not trainable."
        if not self.is_training():
            raise mdp.TrainingFinishedException, \
                  "The training phase has already finished."
        if self.dtype is None:
            self.dtype = forked_node.dtype
        if self.input_dim is None:
            self.input_dim = forked_node.input_dim
        if self.output_dim is None:
            self.output_dim = forked_node.output_dim
        if not self._train_phase_started:
            self._train_phase_started = True
        self._join(forked_node)
            
    ## hook methods, overwrite these ##
    
    def _fork(self):
        """Hook method for forking, to be overridden.
        
        For better inheritance support you should use self.__class__ to create
        new class instances (instead of explicitly referencing the filename).
        """
        raise TrainingPhaseNotParallelException("fork is not implemented " +
                                                "by this node.")
    
    def _join(self, forked_node):
        """Hook method for joining, to be overridden."""
        raise TrainingPhaseNotParallelException("join is not implemented " +
                                                "by this node.")
    

## MDP parallel node implementations ##

class ParallelPCANode(mdp.nodes.PCANode, ParallelNode):
    """Parallel version of MDP PCA node."""
    
    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 svd=False, reduce=False, var_rel=1E-15, var_abs=1E-15, 
                 var_part=None):
        """Initialize the node.
        
        The reduce argument is not supported, since the results may varry for
        different forks.
        """
        super(ParallelPCANode, self).__init__(input_dim=input_dim, 
                                              output_dim=output_dim, 
                                              dtype=dtype,
                                              svd=svd, reduce=reduce, 
                                              var_rel=var_rel, var_abs=var_abs, 
                                              var_part=var_part)
    
    def _fork(self):
        """Fork the node and (if necessary) init the covariance matrices."""
        forked_node = self.__class__(input_dim=self.input_dim, 
                                     output_dim=self.output_dim, 
                                     dtype=self.dtype,
                                     svd=self.svd, reduce=self.reduce, 
                                     var_rel=self.var_rel, var_abs=self.var_abs, 
                                     var_part=self.var_part)
        return forked_node
    
    def _join(self, forked_node):
        """Combine the covariance matrices."""
        if self._cov_mtx._cov_mtx is None:
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
        super(ParallelWhiteningNode, self).__init__(input_dim=input_dim, 
                                                    output_dim=output_dim, 
                                                    dtype=dtype, svd=svd)
    
    def _fork(self):
        """Fork the node and (if necessary) init the covariance matrices."""
        forked_node = self.__class__(input_dim=self.input_dim, 
                                     output_dim=self.output_dim, 
                                     dtype=self.dtype,
                                     svd=self.svd)
        return forked_node
    
    
class ParallelSFANode(mdp.nodes.SFANode, ParallelNode):
    """Parallel version of MDP SFA node."""
    
    def _fork(self):
        """Fork the node and (if necessary) init the covariance matrices."""
        forked_node = self.__class__(input_dim=self.input_dim, 
                                     output_dim=self.output_dim, 
                                     dtype=self.dtype)
        return forked_node
    
    def _join(self, forked_node):
        """Combine the covariance matrices."""
        if self._cov_mtx._cov_mtx is None:
            self.set_dtype(forked_node._cov_mtx._dtype)
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
        forked_node = self.__class__(input_dim=self.input_dim, 
                                     output_dim=self.output_dim, 
                                     dtype=self.dtype)
        return forked_node
    
    
class ParallelFDANode(mdp.nodes.FDANode, ParallelNode):
    
    def _fork(self):
        if self.get_current_train_phase() == 1:
            forked_node = self.copy()
            # reset the variables that might contain data from this train phase
            self._S_W = None
            self._allcov = mdp.utils.CovarianceMatrix(dtype=self.dtype)
        else:
            forked_node = self.__class__(input_dim=self.input_dim, 
                                         output_dim=self.output_dim, 
                                         dtype=self.dtype)
        return forked_node
    
    def _join(self, forked_node):
        if self.get_current_train_phase() == 1:
            if forked_node.get_current_train_phase() != 1:
                msg = ("This node is in training phase 1, but the forked node "
                       "is not.")
                raise JoinParallelNodeException(msg)
            if self._S_W is None:
                self.set_dtype(forked_node._allcov._dtype)
                self._allcov = forked_node._allcov
                self._S_W = forked_node._S_W
            else:
                self._allcov._cov_mtx += forked_node._allcov._cov_mtx
                self._allcov._avg += forked_node._allcov._avg
                self._allcov._tlen += forked_node._allcov._tlen
                self._S_W += forked_node._S_W
        else:
            for lbl in forked_node.means:
                if lbl in self.means:
                    self.means[lbl] += forked_node.means[lbl]
                    self.tlens[lbl] += forked_node.tlens[lbl]
                else:
                    self.means[lbl] = forked_node.means[lbl]
                    self.tlens[lbl] = forked_node.tlens[lbl]
            
            
class ParallelHistogramNode(mdp.nodes.HistogramNode, ParallelNode):
    """Parallel version of the HistogramNode.
    
    Since this node may also be used for parallel execution, the history data
    is dumped in _stop_training to prevent excessive data transfer.    
    """
    
    def __init__(self, hist_fraction=1.0, hist_filename=None, 
                 input_dim=None, dtype=None):
        """Initialize Node.
        
        hist_fraction -- Defines the fraction of the data that is stored
            randomly.
        hist_filename -- Filename for the file to which the data history will
            be pickled after training.
        """
        # TODO: move this to HistogramNode
        self._hist_filename = hist_filename
        super(ParallelHistogramNode, self).__init__(hist_fraction=hist_fraction,  
                                                    input_dim=input_dim,
                                                    dtype=dtype)

    def _fork(self):
        forked_node = self.__class__(hist_fraction=self.hist_fraction,
                                     hist_filename=self._hist_filename,
                                     input_dim=self.input_dim, 
                                     dtype=self.dtype)
        return forked_node
    
    def _join(self, forked_node):
        if (self.data_hist is not None) and (forked_node.data_hist is not None):
            self.data_hist = numx.concatenate([self.data_hist, 
                                            forked_node.data_hist])
        elif forked_node.data_hist != None:
            self.data_hist = forked_node.data_hist
    
    def _stop_training(self):
        """Pickle the data history to file and dump it then."""
        if self._hist_filename:
            pickle_file = open(self._hist_filename, "wb")
            try:
                pickle.dump(self.data_hist, pickle_file, protocol=-1)
            finally:
                pickle_file.close( )
        self.data_hist = None
        super(ParallelHistogramNode, self)._stop_training()


class ParallelAdaptiveCutoffNode(mdp.nodes.AdaptiveCutoffNode,
                                 ParallelHistogramNode):
    """Parallel version of the AdaptiveCutoffNode."""
    
    def __init__(self, lower_cutoff_fraction=None, upper_cutoff_fraction=None, 
                 hist_fraction=1.0, hist_filename=None,
                 input_dim=None, dtype=None):
        """Initialize the node.
        
        lower_cutoff_fraction -- Fraction of data that will be cut off after the
            training phase (assuming the data distribution does not change).
            If set to None no cutoff is performed.
        upper_cutoff_fraction -- Works like lower_cutoff_fraction.
        hist_fraction -- Defines the fraction of the data that is stored for the
            histogram.
        hist_filename -- Filename for the file to which the data history will
            be pickled after training.
        """
        super(ParallelAdaptiveCutoffNode, self).__init__( 
                                lower_cutoff_fraction=lower_cutoff_fraction, 
                                upper_cutoff_fraction=upper_cutoff_fraction, 
                                hist_fraction=hist_fraction,
                                input_dim=input_dim, dtype=dtype)
        self._hist_filename = hist_filename
        
    def _stop_training(self):
        super(ParallelAdaptiveCutoffNode, self)._stop_training()           
