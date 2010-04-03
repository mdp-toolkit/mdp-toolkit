from mdp import Node, numx
import operator

class ClassifierNode(Node):
    """A ClassifierNode can be used for classification tasks that should not interfere
    with the normal execution flow. A Reason for that may be that the labels used
    for classification are not in the normal feature space but in label space.
    """
    def rank(self, x, threshold=None):
        """Returns ordered list with all labels ordered according to prob(x)
        (e.g., [[3 1 2], [2 1 3], ...]).
        
        The optional threshold parameter is used to exclude labels having equal
        or less probability. E.g. threshold=0 excludes all labels with zero probability.
        """
        all_ranking = []
        prob = self.prob(x)
        for p in prob:
            ranking = [(k, v) for k, v in p.items() if v > threshold]
            ranking.sort(key=operator.itemgetter(1))
            ranking = map(operator.itemgetter(0), ranking)
            all_ranking.append(ranking)
        return all_ranking
    
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
        return self._classify(self._refcast(x), *args, **kwargs)
  
    def prob(self, x, *args, **kwargs):
        """Returns the probability for each datapoint and label
        (e.g., [{1:0.1, 2:0.0, 3:0.9}, {1:1.0, 2:0.0, 3:0.0}, ...])

        By default, subclasses should overwrite _prob to implement
        their prob. The docstring of the '_prob' method
        overwrites this docstring.        
        """
        self._pre_execution_checks(x)
        return self._prob(self._refcast(x), *args, **kwargs)


class ClassifierCumulator(ClassifierNode):
    """A ClassifierCumulator is a Node whose training phase simply collects
    all input data and labels. In this way it is possible to easily implement
    batch-mode learning.

    The data is accessible in the attribute 'self.data' after
    the beginning of the '_stop_training' phase. 'self.tlen' contains
    the number of data points collected.
    'self.labels' contains the assigned label to each data point.
    """

    def __init__(self, input_dim = None, output_dim = None, dtype = None):
        super(ClassifierCumulator, self).__init__(input_dim, output_dim, dtype)
        self.data = []
        self.labels = []
        self.tlen = 0

    def _check_train_args(self, x, cl):
        super(ClassifierCumulator, self)._check_train_args(x, cl)
        if (isinstance(cl, (list, tuple, numx.ndarray)) and
            len(cl) != x.shape[0]):
            msg = ("The number of labels must be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)

    def _train(self, x, cl):
        """Cumulate all input data in a one dimensional list."""
        self.tlen += x.shape[0]
        self.data.extend(x.ravel().tolist())

        # if cl is a number, all x's belong to the same class
        if isinstance(cl, (list, tuple, numx.ndarray)):
            pass
        else:
            cl = [cl] * x.shape[0]

        self.labels.extend(cl.ravel().tolist())

    def _stop_training(self, *args, **kwargs):
        """Transform the data and labels lists to array objects and reshape them."""
        self.data = numx.array(self.data, dtype = self.dtype)
        self.data.shape = (self.tlen, self.input_dim)
        self.labels = numx.array(self.labels, dtype = self.dtype)
        self.labels.shape = (self.tlen)
        

