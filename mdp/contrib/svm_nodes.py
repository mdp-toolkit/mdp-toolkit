import mdp
from mdp import numx

class _SVMNode(mdp.Node):

    def __init__(self, input_dim = None, dtype = None):
        self._x = numx.array([]) # train data
        self._cl = numx.array([]) # labels
        self._label_map = {}

        self._norm_labels = numx.array([])

        super(_SVMNode, self).__init__(input_dim, None, dtype)

    def _normalize_labels(self, mode=None):
        """To avoid problems with the algorithms, we normalise the labels to a
        standard layout and take care of the mapping
        """
        if mode == None:
            mode = self._classification_type
        labels = set(self._cl)
        if mode == "dual":
            if len(labels) > 2:
                msg = "In dual mode only two labels can be given"
                raise mdp.NodeException(msg)
            # pop first label and reduce
            if len(labels) > 0:
                l = labels.pop()
                self._label_map[-1] = l
            if len(labels) > 0:
                l = labels.pop()
                self._label_map[1] = l
            else:
                msg = "Training your SVM with only one label is not the most sensible thing to do."
                raise mdp.MDPWarning(msg)
        elif mode == "multi":
            count = 0
            for l in labels:
                self._label_map[count] = l
                count += 1
        else:
            msg = "Remapping mode not known"
            raise mdp.NodeException(msg)

        # now execute the mapping
        try:
            inverted = dict([(v, k) for k, v in self._label_map.iteritems()])
        except TypeError:
            # put more elaborated code here for circumventing this issue
            msg = "Problem inverting. Labels maybe not hashable."
            raise mdp.NodeException(msg)
        norm_labels = map(inverted.get, self._cl)

        if None in self._norm_labels:
            msg = "Error in remapping method"
            raise mdp.NodeException(msg)

        self._norm_labels = numx.array(norm_labels)

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
            msg = ("The number of labels should be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)

    def _train(self, x, cl):
        """Update the internal structures according to the input data 'x'.
        
        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """
        if not len(self._x):
            self._x = x
        else:
            self._x = mdp.numx.append( self._x, x )
        # if cl is a number, all x's belong to the same class
        if isinstance(cl, (list, tuple, numx.ndarray)):
            self._cl = mdp.numx.append( self._cl, cl )
        else:
            cls = [cl] * len(x)
            self._cl = mdp.numx.append( self._cl, cls )
