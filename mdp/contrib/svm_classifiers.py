import mdp
from mdp import numx
from itertools import count

"""General routines and base classes for Support Vector Machine classifiers.

    TODO: Implement some scaling. Either by special Scaling Node or internally.
"""




class _LabelNormalizer(object):
    """This class provides a transparent mapping from arbitrary labels
    to a set of well-defined integers.
    
    TODO: This could actually be a node.
    TODO: Needs more refinement. E.g. could automatically round labels to +1, -1
    """
    def __init__(self, labels, mode=None):
        if mode is None:
            mode = "id"
        if mode == "id":
            self.normalize = self._id
            self.revert = self._id
            return
        
        self._mode = mode
        self._labels = set(labels)
        self._mapping = {}
        self._inverse = {}
        if mode == "dual":
            if len(self._labels) > 2:
                msg = "In dual mode only two labels can be given"
                raise mdp.NodeException(msg)
            t_label_norm = zip(self._labels, [1, -1])
            self._set_label_dicts(t_label_norm)
        elif mode == "multi":
            # enumerate from zero to len
            t_label_norm = zip(self._labels, count())
            self._set_label_dicts(t_label_norm)
        else:
            msg = "Remapping mode not known"
            raise mdp.NodeException(msg)
    
    def _set_label_dicts(self, t_label_norm):
        self._mapping = dict(t_label_norm)
        self._inverse = dict((norm, label) for label, norm in t_label_norm)
        
        # check that neither original nor normalised labels have occured more than once
        if not (len(self._mapping) == len(t_label_norm) == len(self._inverse)):
            msg = "Error in label normalisation."
            raise mdp.NodeException(msg) 
    
    def normalize(self, labels):
        return map(self._mapping.get, labels)
    
    def revert(self, norm_labels):
        return map(self._inverse.get, norm_labels)
    
    def _id(self, labels):
        return labels


class _SVMClassifier(mdp.ClassifierNode):
    """Base class for the SVM classifier nodes."""

    def __init__(self, input_dim = None, dtype = None):
        self._in_features = numx.array([]) # train data
        self._in_labels = numx.array([]) # labels

        self.normalizer = None

        super(_SVMClassifier, self).__init__(input_dim, None, dtype)

    def is_invertible(self):
        return False

    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n

    def _set_output_dim(self, n):
        msg = "Output dim cannot be set explicitly!"
        raise mdp.NodeException(msg)

    def _check_train_args(self, x, cl):
        if (isinstance(cl, (list, tuple, numx.ndarray)) and
            len(cl) != x.shape[0]):
            msg = ("The number of labels must be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)

    def _append_data(self, x, cl):
        """Updates self._in_featurs and self._in_labels with appended data from x and cl.
        """
        if not len(self._in_features):
            self._in_features = x
        else:
            self._in_features = numx.concatenate( (self._in_features, x) )
        # if cl is a number, all x's belong to the same class
        if isinstance(cl, (list, tuple, numx.ndarray)):
            self._in_labels = numx.concatenate( (self._in_labels, cl) )
        else:
            cls = [cl] * len(x)
            self._in_labels = numx.concatenate( (self._in_labels, cls) )

    def _train(self, x, cl):
        """Update the internal structures according to the input data 'x'.
        
        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """
        self._append_data(x, cl)
        
