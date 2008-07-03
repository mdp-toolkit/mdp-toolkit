import cPickle as _cPickle
import warnings as _warnings
import mdp

# import numeric module (scipy, Numeric or numarray)
numx = mdp.numx

class NodeException(mdp.MDPException):
    """Base class for exceptions in Node subclasses."""
    pass

class TrainingException(NodeException):
    """Base class for exceptions in the training phase."""
    pass

class TrainingFinishedException(TrainingException):
    """Raised when the 'train' function is called although the
    training phase is closed."""
    pass

class IsNotTrainableException(TrainingException):
    """Raised when the 'train' function is called although the
    node is not trainable."""
    pass

class IsNotInvertibleException(NodeException):
    """Raised when the 'inverse' function is called although the
    node is not invertible."""
    pass

class Node(object):
    """
    Node is the basic unit in MDP and it represents a data processing
    element, like for example a learning algorithm, a filter, a
    visualization step, etc. Each Node can have one or more training
    phases, during which the internal structures are learned from training
    data (e.g. the weights of a neural network are adapted or the
    covariance matrix is estimated) and an execution phase, where new data
    can be processed forwards (by processing the data through the node) or
    backwards (by applying the inverse of the transformation computed by
    the node if defined). The Node class is designed to make the
    implementation of new algorithms easy and intuitive, for example by
    setting automatically input and output dimension and by casting the
    data to match the numerical type (e.g. float or double) of the
    internal structures. Node was designed to be applied to arbitrarily
    long sets of data: the internal structures can be updated
    incrementally by sending chunks of the input data (this is equivalent
    to online learning if the chunks consists of single observations, or
    to batch learning if the whole data is sent in a single chunk).

    Node subclasses should take care of overwriting (if necessary)
    the functions is_trainable, _train, _stop_training, _execute,
    is_invertible, _inverse, _get_train_seq, and _get_supported_dtypes.
    If you need to overwrite the getters and setters of the
    node's properties refer to the docstring of get/set_input_dim,
    get/set_output_dim, and get/set_dtype."""

    def __init__(self, input_dim = None, output_dim = None, dtype = None):
        """If the input dimension and the output dimension are
        unspecified, they will be set when the 'train' or 'execute'
        function is called for the first time.
        If dtype is unspecified, it will be inherited from the data
        it receives at the first call of 'train' or 'execute'. Every subclass
        must take care of up- or down-casting the internal
        structures to match this argument (use _refcast private
        method when possible).
        """
        # initialize basic attributes
        self._input_dim = None
        self._output_dim = None
        self._dtype = None
        # call set functions for properties
        self.set_input_dim(input_dim)
        self.set_output_dim(output_dim)
        self.set_dtype(dtype)
 
        # skip the training phase if the node is not trainable
        if not self.is_trainable():
            self._training = False
            self._train_phase = -1
            self._train_phase_started = False
        else:
            # this var stores at which point in the training sequence we are
            self._train_phase = 0          
            # this var is False if the training of the current phase hasn't
            #  started yet, True otherwise
            self._train_phase_started = False
            # this var is False if the complete training is finished
            self._training = True

    ### properties

    def get_input_dim(self):
        """Return input dimensions."""
        return self._input_dim

    def set_input_dim(self, n):
        """Set input dimensions.
        Performs sanity checks and then calls self._set_input_dim(n), which
        is responsible for setting the internal attribute self._input_dim.
        Note that subclasses should overwrite self._set_input_dim
        when needed."""
        if n is None:
            pass
        elif (self._input_dim is not None) and (self._input_dim !=  n):
            msg = "Input dim are set already (%d) "%(self.input_dim)+\
                  "(%d given)!"%n
            raise NodeException, msg
        else:
            self._set_input_dim(n)

    def _set_input_dim(self, n):
        self._input_dim = n

    input_dim = property(get_input_dim,
                         set_input_dim,
                         doc = "Input dimensions")

    def get_output_dim(self):
        """Return output dimensions."""
        return self._output_dim

    def set_output_dim(self, n):
        """Set output dimensions.
        Performs sanity checks and then calls self._set_output_dim(n), which
        is responsible for setting the internal attribute self._output_dim.
        Note that subclasses should overwrite self._set_output_dim
        when needed."""
        if n is None:
            pass
        elif (self._output_dim is not None) and (self._output_dim != n):
            msg = "Output dim are set already (%d) "%(self.output_dim)+\
                  "(%d given)!"%(n)
            raise NodeException, msg
        else:
            self._set_output_dim(n)

    def _set_output_dim(self, n):
        self._output_dim = n

    output_dim = property(get_output_dim,
                          set_output_dim,
                          doc = "Output dimensions")

    def get_dtype(self):
        """Return dtype."""
        return self._dtype
    
    def set_dtype(self, t):
        """Set Node's internal structures dtype.
        Performs sanity checks and then calls self._set_dtype(n), which
        is responsible for setting the internal attribute self._dtype.
        Note that subclasses should overwrite self._set_dtype
        when needed."""
        if t is None: return
        t = numx.dtype(t)
        if (self._dtype is not None) and (self._dtype != t):
            errstr = "dtype is already set to '%s' " % (self.dtype.name)+\
                     "('%s' given)!"%t
            raise NodeException, errstr
        elif t not in self.get_supported_dtypes():
            errstr = "\ndtype '%s' is not supported.\n" % t.name+ \
                      "Supported dtypes: %s" \
                      %([mdp.numx.dtype(t).name for t in
                         self.get_supported_dtypes()])
            raise NodeException, errstr
        else:
            self._set_dtype(t)

    def _set_dtype(self, t):
        self._dtype = t
        
    dtype = property(get_dtype,
                     set_dtype,
                     doc = "dtype")

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node.
        The types can be specified in any format allowed by numpy.dtype."""
        return mdp.utils.get_dtypes('All')

    def get_supported_dtypes(self):
        """Return dtypes supported by the node as a list of numpy.dtype
        objects.
        Note that subclasses should overwrite self._get_supported_dtypes
        when needed."""
        return [numx.dtype(t) for t in self._get_supported_dtypes()]

    supported_dtypes = property(get_supported_dtypes,
                                doc = "Supported dtypes")

    _train_seq = property(lambda self: self._get_train_seq(),
                          doc = "List of tuples: [(training-phase1, " +\
                          "stop-training-phase1), (training-phase2, " +\
                          "stop_training-phase2), ... ].\n" +\
                          " By default _train_seq = [(self._train," +\
                          " self._stop_training]")

    def _get_train_seq(self):
        return [(self._train, self._stop_training)]

    ### Node states
    def is_training(self):
        """Return True if the node is in the training phase,
        False otherwise."""
        return self._training

    def get_current_train_phase(self):
        """Return the index of the current training phase. The training phases
        are defined in the list self._train_seq."""
        return self._train_phase

    def get_remaining_train_phase(self):
        """Return the number of training phases still to accomplish."""
        return len(self._train_seq) - self._train_phase

    ### Node capabilities
    def is_trainable(self):
        """Return True if the node can be trained, False otherwise."""
        return True

    def is_invertible(self):
        """Return True if the node can be inverted, False otherwise."""
        return True

    ### check functions
    def _check_input(self, x):
        # check input rank
        if not x.ndim == 2:
            error_str = "x has rank %d, should be 2"\
                        %(x.ndim)
            raise NodeException, error_str

        # set the input dimension if necessary
        if self.input_dim is None:
            self.input_dim = x.shape[1]

        # set the dtype if necessary
        if self.dtype is None:
            self.dtype = x.dtype

        # check the input dimension
        if not x.shape[1] == self.input_dim:
            error_str = "x has dimension %d, should be %d" \
                        % (x.shape[1], self.input_dim)
            raise NodeException, error_str

        if x.shape[0] == 0:
            error_str = "x must have at least one observation (zero given)"
            raise NodeException, error_str
        
    def _check_output(self, y):
        # check output rank
        if not y.ndim == 2:
            error_str = "y has rank %d, should be 2"\
                        %(y.ndim)
            raise NodeException, error_str

        # check the output dimension
        if not y.shape[1] == self.output_dim:
            error_str = "y has dimension %d, should be %d" \
                        % (y.shape[1], self.output_dim)
            raise NodeException, error_str

    def _if_training_stop_training(self):
        if self.is_training():
            self.stop_training()
            # if there is some training phases left
            # we shouldn't be here!
            if self.get_remaining_train_phase() > 0:
                raise TrainingException, \
                      "The training phases are not completed yet."

    def _pre_execution_checks(self, x):
        """This method contains all pre-execution checks.
        It can be used when a subclass defines multiple execution methods."""
        
        self._if_training_stop_training()
        
        # control the dimension x
        self._check_input(x)

        # set the output dimension if necessary
        if self.output_dim is None:
            self.output_dim = self.input_dim

    def _pre_inversion_checks(self, y):
        """This method contains all pre-inversion checks.
        It can be used when a subclass defines multiple inversion methods."""
        
        if not self.is_invertible():
            raise IsNotInvertibleException, "This node is not invertible."

        self._if_training_stop_training()

        # set the output dimension if necessary
        if self.output_dim is None:
            # if the input_dim is not defined, raise an exception
            if self.input_dim is None:
                errstr = "Number of input dimensions undefined. Inversion"+\
                         "not possible."
                raise NodeException, errstr
            self.output_dim = self.input_dim
        
        # control the dimension of y
        self._check_output(y)

    def _check_train_args(self, x, *args, **kwargs):
        # implemented by subclasses if needed
        pass

    ### casting helper functions

    def _refcast(self, x):
        """Helper function to cast arrays to the internal dtype."""
        return mdp.utils.refcast(x, self.dtype)
    
    ### Methods to be implemented by the user

    # this are the methods the user has to overwrite
    # they receive the data already casted to the correct type
    
    def _train(self, x):
        if self.is_trainable():
            raise NotImplementedError

    def _stop_training(self, *args, **kwargs):
        pass

    def _execute(self, x):
        return x
        
    def _inverse(self, x):
        if self.is_invertible():
            return x

    ### User interface to the overwritten methods
    
    def train(self, x, *args, **kwargs):
        """Update the internal structures according to the input data 'x'.
        
        'x' is a matrix having different variables on different columns
        and observations on the rows.

        Be default, subclasses should overwrite _train to implement their
        training phase. This method can be overwritten to redefine its
        docstring. For example:

        def train(self, x, arg1, arg2):
            ""My training method. arg1 is the first argument, arg2 the second""
            super(MyNode, self).train(x, arg1, arg2)
        """

        if not self.is_trainable():
            raise IsNotTrainableException, "This node is not trainable."

        if not self.is_training():
            raise TrainingFinishedException, \
                  "The training phase has already finished."

        self._check_input(x)
        self._check_train_args(x, *args, **kwargs)        
        
        self._train_phase_started = True
        self._train_seq[self._train_phase][0](self._refcast(x), *args,**kwargs)

    def stop_training(self, *args, **kwargs):
        """Stop the training phase.
        Be default, subclasses should overwrite _stop_Training to implement
        their stop-training."""
        if self.is_training() and self._train_phase_started == False:
            raise TrainingException, \
                  "The node has not been trained."
        
        if not self.is_training():
            raise TrainingFinishedException, \
                  "The training phase has already finished."

        # close the current phase.
        self._train_seq[self._train_phase][1](*args, **kwargs)
        self._train_phase += 1
        self._train_phase_started = False
        # check if we have some training phase left
        if self.get_remaining_train_phase() == 0:
            self._training = False

    def execute(self, x, *args, **kargs):
        """Process the data contained in 'x'.
        
        If the object is still in the training phase, the function
        'stop_training' will be called.
        'x' is a matrix having different variables on different columns
        and observations on the rows.
        
        Subclasses should overwrite _execute to implement their
        execution phase. This method can be overwritten to redefine its
        docstring. For example:

        def execute(self, x, arg1, karg2=0.):
            ""My execute method. arg1 is the first argument, karg2 the second""
            super(MyNode, self).execute(x, arg1, karg2=karg2)
        """
        self._pre_execution_checks(x)
        return self._execute(self._refcast(x), *args, **kargs)

    def inverse(self, y, *args, **kargs):
        """Invert 'y'.
        
        If the node is invertible, compute the input x such that
        y = execute(x).
        
        Subclasses should overwrite _inverse to implement their
        inverse function. This method can be overwritten to redefine its
        docstring. For example:

        def inverse(self, x, arg1, karg2=0.):
            ""My inverse method. arg1 is the first argument, karg2 the second""
            super(MyNode, self).inverse(x, arg1, karg2=karg2)
        """
        self._pre_inversion_checks(y)
        return self._inverse(self._refcast(y), *args, **kargs)

    def __call__(self, x, *args, **kargs):
        """Calling an instance if Node is equivalent to call
        its 'execute' method."""
        return self.execute(x, *args, **kargs)

    ###### string representation
    
    def __str__(self):
        return str(type(self).__name__)
    
    def __repr__(self):
        # print input_dim, output_dim, dtype 
        name = type(self).__name__
        inp = "input_dim=%s"%str(self.input_dim)
        out = "output_dim=%s"%str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" %self.dtype.name
        args = ', '.join((inp, out, typ))
        return name+'('+args+')'

    def copy(self, protocol = -1):
        """Return a deep copy of the node.
        Protocol is the pickle protocol."""
        as_str = _cPickle.dumps(self, protocol)
        return _cPickle.loads(as_str)

    def save(self, filename, protocol = -1):
        """Save a pickled representation of the node to 'filename'.
        If 'filename' is None, return a string.

        Note: the pickled Node is not guaranteed to be upward or
        backward compatible."""
        if filename is None:
            return _cPickle.dumps(self, protocol)
        else:
            # if protocol != 0 open the file in binary mode
            if protocol != 0:
                mode = 'wb'
            else:
                mode = 'w'
            flh = open(filename, mode)
            _cPickle.dump(self, flh, protocol)
            flh.close()

class Cumulator(Node):
    """A Cumulator is a Node whose training phase simply cumulates
    all input data.
    In this way it is possible to easily implement batch-mode learning.
    """

    def __init__(self, input_dim = None, output_dim = None, dtype = None):
        super(Cumulator, self).__init__(input_dim, output_dim, dtype)
        self.data = []
        self.tlen = 0

    def _train(self, x):
        """Cumulate all imput data in a one dimensional list.
        """
        self.tlen += x.shape[0]
        self.data.extend(x.ravel().tolist())

    def _stop_training(self):
        """Transform the data list to an array object and reshape it.
        """
        self._training = False
        self.data = numx.array(self.data, dtype = self.dtype)
        self.data.shape = (self.tlen, self.input_dim)
