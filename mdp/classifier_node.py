import mdp

class ClassifierNode(mdp.Node):
    """A ClassifierNode can be used for classification tasks that should not interfere
    with the normal execution flow. A Reason for that may be that the labels used
    for classification are not in the normal feature space but in label space.
    """
    def rank(self, x, threshold = None):
        """Returns ordered list with all labels ordered according to prob(x)
        (e.g., [3 1 2])
        """
        p = prob(x)
        ranking = [(k,v) for k,v in p if v > threshold]
        ranking.sort(cmp=lambda x, y: cmp(x[1], y[1]))
        return ranking
    
    def _classify(self, x, *args, **kargs):
        raise NotImplementedError
    
    def _prob(self, x, *args, **kargs):
        raise NotImplementedError
    
    ### User interface to the overwritten methods
    
    def classify(self, x, *args, **kwargs):
        """Returns an array with best labels.
        
        By default, subclasses should overwrite _classify to implement
        their classify. The docstring of the '_classify' method
        overwrites this docstring.
        """
        self._pre_execution_checks(x)
        return self._classify(x, *args, **kwargs)
  
    def prob(self, x, *args, **kwargs):
        """Returns the probability for each datapoint and label
        (e.g., {1:0.1, 2:0.0, 3:0.9})

        By default, subclasses should overwrite _prob to implement
        their prob. The docstring of the '_prob' method
        overwrites this docstring.        
        """
        self._pre_execution_checks(x)
        return self._prob(x, *args, **kwargs)
        