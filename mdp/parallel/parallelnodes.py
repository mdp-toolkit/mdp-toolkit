"""
Module for MDP Nodes that support parallel training.

This module contains both the parallel base class and some parallel 
implementations of MDP nodes. Note that such ParallelNodes are only needed for 
training, parallel execution works with any Node that can be pickled.
"""

# WARNING: There is a problem with unpickled arrays in NumPy < 1.1.x, see
# http://projects.scipy.org/scipy/numpy/ticket/551
# To circumvent this, you can use a copy() of all unpickled arrays. 

import inspect

import mdp
from mdp import numx


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


class ParallelExtensionNode(mdp.ExtensionNode, mdp.Node):
    """Base class for parallel trainable MDP nodes.
    
    With the fork method new node instances are created which can then be
    trained. With the join method the trained instances are then merged back
    into a single node instance.
    
    This class defines default methods which raise a
    TrainingPhaseNotParallelException exception.
    """
    
    extension_name = "parallel" 
    
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
        """Hook method for forking with default implementation.
        
        Overwrite this method for nodes that can be parallelized.
        You can use _default_fork, if that is compatible with your node class,
        typically the hard part is the joining.
        """
        raise TrainingPhaseNotParallelException("fork is not implemented " +
                                                "by this node (%s)" %
                                                str(self.__class__))
    
    def _join(self, forked_node):
        """Hook method for joining, to be overridden."""
        raise TrainingPhaseNotParallelException("join is not implemented " +
                                                "by this node (%s)" %
                                                str(self.__class__))
    
    ## helper methods ##
        
    def _default_fork(self):
        """Default implementation of _fork.
        
        It uses introspection to determine the init kwargs and tries to fill
        them with public attributes. These kwargs are then used to instanciate
        self.__class__ to create the fork instance.
        
        So you can use this method if all the required keys are also public
        attributes or have a single underscore in front.
        """
        args, varargs, varkw, defaults = inspect.getargspec(self.__init__)
        args.remove("self")
        if defaults:
            non_default_keys = args[:-len(defaults)]
        else:
            non_default_keys = []
        kwargs = dict((key, getattr(self, key))
                      for key in args if hasattr(self, key))
        # look for the key with an underscore in front
        for key in kwargs:
            args.remove(key)
        under_kwargs = dict((key, getattr(self, '_' + key))
                            for key in args if hasattr(self, '_' + key))
        for key in under_kwargs:
            args.remove(key)
        kwargs.update(under_kwargs)
        # check that all the keys without default arguments are covered
        if non_default_keys:
            missing_defaults = set(non_default_keys) & set(args)
            if missing_defaults:
                err = ("could not find attributes for init arguments %s" %
                       str(missing_defaults))
                raise TrainingPhaseNotParallelException(err)
        # create new instance
        return self.__class__(**kwargs)
    

## MDP parallel node implementations ##

class ParallelPCANode(ParallelExtensionNode, mdp.nodes.PCANode):
    """Parallel version of MDP PCA node."""
    
    def _fork(self):
        return self._default_fork()
    
    def _join(self, forked_node):
        """Combine the covariance matrices."""
        if self._cov_mtx._cov_mtx is None:
            self.set_dtype(self._cov_mtx._dtype)
            self._cov_mtx = forked_node._cov_mtx
        else:
            self._cov_mtx._cov_mtx += forked_node._cov_mtx._cov_mtx
            self._cov_mtx._avg += forked_node._cov_mtx._avg
            self._cov_mtx._tlen += forked_node._cov_mtx._tlen
            
            
class ParallelSFANode(ParallelExtensionNode, mdp.nodes.SFANode):
    """Parallel version of MDP SFA node."""
    
    def _fork(self):
        return self._default_fork()
    
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
            
            
class ParallelFDANode(ParallelExtensionNode, mdp.nodes.FDANode):
    
    def _fork(self):
        if self.get_current_train_phase() == 1:
            forked_node = self.copy()
            # reset the variables that might contain data from this train phase
            forked_node._S_W = None
            forked_node._allcov = mdp.utils.CovarianceMatrix(dtype=self.dtype)
        else:
            forked_node = self._default_fork()
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
            
            
class ParallelHistogramNode(ParallelExtensionNode, mdp.nodes.HistogramNode):
    """Parallel version of the HistogramNode."""
    
    def _fork(self):
        return self._default_fork()
    
    def _join(self, forked_node):
        if (self.data_hist is not None) and (forked_node.data_hist is not None):
            self.data_hist = numx.concatenate([self.data_hist, 
                                            forked_node.data_hist])
        elif forked_node.data_hist != None:
            self.data_hist = forked_node.data_hist
    