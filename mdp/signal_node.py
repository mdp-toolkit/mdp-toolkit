import cPickle as _cPickle
import inspect as _inspect
import types

import mdp
from mdp import numx


class NodeException(mdp.MDPException):
    """Base class for exceptions in Node subclasses."""
    pass

class TrainingException(NodeException):
    """Base class for exceptions in the training phase."""
    pass

class TrainingFinishedException(TrainingException):
    """Raised when the 'train' method is called although the
    training phase is closed."""
    pass

class IsNotTrainableException(TrainingException):
    """Raised when the 'train' method is called although the
    node is not trainable."""
    pass

class IsNotInvertibleException(NodeException):
    """Raised when the 'inverse' method is called although the
    node is not invertible."""
    pass


class NodeMetaclass(type):
    """This Metaclass is meant to overwrite doc strings of methods like
    execute, stop_training, inverse with the ones defined in the corresponding
    private methods _execute, _stop_training, _inverse, etc...

    This makes it possible for subclasses of Node to document the usage
    of public methods, without the need to overwrite the ancestor's methods.
    """

    # methods that can overwrite docs:
    DOC_METHODS = ['_train', '_stop_training', '_execute', '_inverse']
    
    def __new__(cls, classname, bases, members):
        # select private methods that can overwrite the docstring
        for privname in cls.DOC_METHODS:
            if privname in members:
                # the private method is present in the class
                # inspect the private method
                priv_info = cls._get_infodict(members[privname])
                # if the docstring is empty, don't overwrite it
                if not priv_info['doc']:
                    continue
                # get the name of the corresponding public method
                pubname = privname[1:]
                # if public method has been overwritten in this
                # subclass, keep it
                if pubname in members:
                    continue
                # look for public method by same name in ancestors
                for base in bases:
                    ancestor = base.__dict__
                    if pubname in ancestor:
                        # we found a method by the same name in the ancestor
                        # add the class a wrapper of the ancestor public method
                        # with docs, argument defaults, and signature of the
                        # private method and name of the public method.
                        priv_info['name'] = pubname
                        # preserve signature, this is used in binet
                        pub_info = cls._get_infodict(ancestor[pubname])
                        priv_info['signature'] = pub_info['signature']
                        priv_info['argnames'] = pub_info['argnames']
                        priv_info['defaults'] = pub_info['defaults']
                        members[pubname] = cls._wrap_func(ancestor[pubname], 
                                                          priv_info)
                        break
        return super(NodeMetaclass, NodeMetaclass).__new__(cls, classname,
                                                           bases, members)

    # The next two functions (originally called get_info, wrapper)
    # are adapted versions of functions in the
    # decorator module by Michele Simionato
    # Version: 2.3.1 (25 July 2008)
    # Download page: http://pypi.python.org/pypi/decorator
    
    @staticmethod
    def _get_infodict(func):
        """
        Returns an info dictionary containing:
        - name (the name of the function : str)
        - argnames (the names of the arguments : list)
        - defaults (the values of the default arguments : tuple)
        - signature (the signature : str)
        - doc (the docstring : str)
        - module (the module name : str)
        - dict (the function __dict__ : str)
        
        >>> def f(self, x=1, y=2, *args, **kw): pass
    
        >>> info = getinfo(f)
    
        >>> info["name"]
        'f'
        >>> info["argnames"]
        ['self', 'x', 'y', 'args', 'kw']
        
        >>> info["defaults"]
        (1, 2)
    
        >>> info["signature"]
        'self, x, y, *args, **kw'
        """
        regargs, varargs, varkwargs, defaults = _inspect.getargspec(func)
        argnames = list(regargs)
        if varargs:
            argnames.append(varargs)
        if varkwargs:
            argnames.append(varkwargs)
        signature = _inspect.formatargspec(regargs,
                                           varargs,
                                           varkwargs,
                                           defaults,
                                           formatvalue=lambda value: "")[1:-1]
        return dict(name=func.__name__, argnames=argnames, signature=signature,
                    defaults = func.func_defaults, doc=func.__doc__,
                    module=func.__module__, dict=func.__dict__,
                    globals=func.func_globals, closure=func.func_closure)
    
    @staticmethod
    def _wrap_func(original_func, wrapper_infodict):
        """Return a wrapped version of func.
        
        original_func -- The function to be wrapped.
        wrapper_infodict -- The infodict to be used for constructing the
            wrapper.
        """
        src = ("lambda %(signature)s: _original_func_(%(signature)s)" %
               wrapper_infodict)
        wrapped_func = eval(src, dict(_original_func_=original_func))
        wrapped_func.__name__ = wrapper_infodict['name']
        wrapped_func.__doc__ = wrapper_infodict['doc']
        wrapped_func.__module__ = wrapper_infodict['module']
        wrapped_func.__dict__.update(wrapper_infodict['dict'])
        wrapped_func.func_defaults = wrapper_infodict['defaults']
        wrapped_func.undecorated = wrapper_infodict
        return wrapped_func

class Node(object):
    """A 'Node' is the basic building block of an MDP application.

    It represents a data processing element, like for example a learning
    algorithm, a data filter, or a visualization step.
    Each node can have one or more training phases, during which the
    internal structures are learned from training data (e.g. the weights
    of a neural network are adapted or the covariance matrix is estimated)
    and an execution phase, where new data can be processed forwards (by
    processing the data through the node) or backwards (by applying the
    inverse of the transformation computed by the node if defined).

    Nodes have been designed to be applied to arbitrarily long sets of data:
    if the underlying algorithms supports it, the internal structures can
    be updated incrementally by sending multiple batches of data (this is
    equivalent to online learning if the chunks consists of single
    observations, or to batch learning if the whole data is sent in a
    single chunk). It is thus possible to perform computations on amounts
    of data that would not fit into memory or to generate data on-the-fly.

    A 'Node' also defines some utility methods, like for example
    'copy' and 'save', that return an exact copy of a node and save it
    in a file, respectively. Additional methods may be present, depending
    on the algorithm.

    Node subclasses should take care of overwriting (if necessary)
    the functions is_trainable, _train, _stop_training, _execute,
    is_invertible, _inverse, _get_train_seq, and _get_supported_dtypes.
    If you need to overwrite the getters and setters of the
    node's properties refer to the docstring of get/set_input_dim,
    get/set_output_dim, and get/set_dtype.
    """

    __metaclass__ = NodeMetaclass

    def __init__(self, input_dim = None, output_dim = None, dtype = None):
        """If the input dimension and the output dimension are
        unspecified, they will be set when the 'train' or 'execute'
        method is called for the first time.
        If dtype is unspecified, it will be inherited from the data
        it receives at the first call of 'train' or 'execute'.

        Every subclass must take care of up- or down-casting the internal
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
        
        Perform sanity checks and then calls self._set_input_dim(n), which
        is responsible for setting the internal attribute self._input_dim.
        Note that subclasses should overwrite self._set_input_dim
        when needed."""
        if n is None:
            pass
        elif (self._input_dim is not None) and (self._input_dim !=  n):
            msg = ("Input dim are set already (%d) "
                   "(%d given)!" % (self.input_dim, n))
            raise NodeException(msg)
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
        Perform sanity checks and then calls self._set_output_dim(n), which
        is responsible for setting the internal attribute self._output_dim.
        Note that subclasses should overwrite self._set_output_dim
        when needed."""
        if n is None:
            pass
        elif (self._output_dim is not None) and (self._output_dim != n):
            msg = ("Output dim are set already (%d) "
                   "(%d given)!" % (self.output_dim, n))
            raise NodeException(msg)
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
        """Set internal structures' dtype.
        Perform sanity checks and then calls self._set_dtype(n), which
        is responsible for setting the internal attribute self._dtype.
        Note that subclasses should overwrite self._set_dtype
        when needed."""
        if t is None:
            return
        t = numx.dtype(t)
        if (self._dtype is not None) and (self._dtype != t):
            errstr = ("dtype is already set to '%s' "
                      "('%s' given)!" % (t, self.dtype.name)) 
            raise NodeException(errstr)
        elif t not in self.get_supported_dtypes():
            errstr = ("\ndtype '%s' is not supported.\n"
                      "Supported dtypes: %s" % ( t.name,
                                                 [numx.dtype(t).name for t in
                                                  self.get_supported_dtypes()]))
            raise NodeException(errstr)
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
                          doc = "List of tuples: [(training-phase1, "
                          "stop-training-phase1), (training-phase2, "
                          "stop_training-phase2), ... ].\n"
                          " By default _train_seq = [(self._train,"
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
        """Return the number of training phases still to accomplish.
        
        If the node is not trainable then the return value is 0.
        """
        if self.is_trainable():
            return len(self._train_seq) - self._train_phase
        else:
            return 0

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
            error_str = "x has rank %d, should be 2" % (x.ndim)
            raise NodeException(error_str)

        # set the input dimension if necessary
        if self.input_dim is None:
            self.input_dim = x.shape[1]

        # set the dtype if necessary
        if self.dtype is None:
            self.dtype = x.dtype

        # check the input dimension
        if not x.shape[1] == self.input_dim:
            error_str = "x has dimension %d, should be %d" % (x.shape[1],
                                                              self.input_dim)
            raise NodeException(error_str)

        if x.shape[0] == 0:
            error_str = "x must have at least one observation (zero given)"
            raise NodeException(error_str)
        
    def _check_output(self, y):
        # check output rank
        if not y.ndim == 2:
            error_str = "y has rank %d, should be 2" % (y.ndim)
            raise NodeException(error_str)

        # check the output dimension
        if not y.shape[1] == self.output_dim:
            error_str = "y has dimension %d, should be %d" % (y.shape[1],
                                                              self.output_dim)
            raise NodeException(error_str)

    def _if_training_stop_training(self):
        if self.is_training():
            self.stop_training()
            # if there is some training phases left we shouldn't be here!
            if self.get_remaining_train_phase() > 0:
                error_str = "The training phases are not completed yet."
                raise TrainingException(error_str)

    def _pre_execution_checks(self, x):
        """This method contains all pre-execution checks.
        It can be used when a subclass defines multiple execution methods."""

        # if training has not started yet, assume we want to train the node
        if (self.get_current_train_phase() == 0 and
            not self._train_phase_started):
            while True:
                self.train(x)
                if self.get_remaining_train_phase() > 1:
                    self.stop_training()
                else:
                    break
        
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
            raise IsNotInvertibleException("This node is not invertible.")

        self._if_training_stop_training()

        # set the output dimension if necessary
        if self.output_dim is None:
            # if the input_dim is not defined, raise an exception
            if self.input_dim is None:
                errstr = ("Number of input dimensions undefined. Inversion"
                          "not possible.")
                raise NodeException(errstr)
            self.output_dim = self.input_dim
        
        # control the dimension of y
        self._check_output(y)

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

    def _check_train_args(self, x, *args, **kwargs):
        # implemented by subclasses if needed
        pass

    ### User interface to the overwritten methods
    
    def train(self, x, *args, **kwargs):
        """Update the internal structures according to the input data 'x'.
        
        'x' is a matrix having different variables on different columns
        and observations on the rows.

        By default, subclasses should overwrite _train to implement their
        training phase. The docstring of the '_train' method overwrites this
        docstring.
        """

        if not self.is_trainable():
            raise IsNotTrainableException("This node is not trainable.")

        if not self.is_training():
            err_str = "The training phase has already finished."
            raise TrainingFinishedException(err_str)

        self._check_input(x)
        self._check_train_args(x, *args, **kwargs)        
        
        self._train_phase_started = True
        self._train_seq[self._train_phase][0](self._refcast(x), *args, **kwargs)

    def stop_training(self, *args, **kwargs):
        """Stop the training phase.

        By default, subclasses should overwrite _stop_training to implement
        their stop-training. The docstring of the '_stop_training' method
        overwrites this docstring.
        """
        if self.is_training() and self._train_phase_started == False:
            raise TrainingException("The node has not been trained.")
        
        if not self.is_training():
            err_str = "The training phase has already finished."
            raise TrainingFinishedException(err_str)

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
        
        By default, subclasses should overwrite _execute to implement
        their execution phase. The docstring of the '_execute' method
        overwrites this docstring.
        """
        self._pre_execution_checks(x)
        return self._execute(self._refcast(x), *args, **kargs)

    def inverse(self, y, *args, **kargs):
        """Invert 'y'.
        
        If the node is invertible, compute the input x such that
        y = execute(x).
        
        By default, subclasses should overwrite _inverse to implement
        their inverse function. The docstring of the '_inverse' method
        overwrites this docstring.
        """
        self._pre_inversion_checks(y)
        return self._inverse(self._refcast(y), *args, **kargs)

    def __call__(self, x, *args, **kargs):
        """Calling an instance of Node is equivalent to call
        its 'execute' method."""
        return self.execute(x, *args, **kargs)

    ###### adding nodes returns flows

    def __add__(self, other):
        # check other is a node
        if isinstance(other, Node):
            return mdp.Flow([self, other])
        elif isinstance(other, mdp.Flow):
            flow_copy = other.copy()
            flow_copy.insert(0, self)
            return flow_copy.copy()
        else:
            err_str = ('can only concatenate node'
                       ' (not \'%s\') to node' % (type(other).__name__) )
            raise TypeError(err_str)
        
    ###### string representation
    
    def __str__(self):
        return str(type(self).__name__)
    
    def __repr__(self):
        # print input_dim, output_dim, dtype 
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" % self.dtype.name
        args = ', '.join((inp, out, typ))
        return name+'('+args+')'

    def copy(self, protocol = -1):
        """Return a deep copy of the node.
        Protocol is the pickle protocol."""
        as_str = _cPickle.dumps(self, protocol)
        return _cPickle.loads(as_str)

    def save(self, filename, protocol = -1):
        """Save a pickled serialization of the node to 'filename'.
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
    """A Cumulator is a Node whose training phase simply collects
    all input data. In this way it is possible to easily implement
    batch-mode learning.

    The data is accessible in the attribute 'self.data' after
    the beginning of the '_stop_training' phase. 'self.tlen' contains
    the number of data points collected.
    """

    def __init__(self, input_dim = None, output_dim = None, dtype = None):
        super(Cumulator, self).__init__(input_dim, output_dim, dtype)
        self.data = []
        self.tlen = 0

    def _train(self, x):
        """Cumulate all input data in a one dimensional list."""
        self.tlen += x.shape[0]
        self.data.extend(x.ravel().tolist())

    def _stop_training(self, *args, **kwargs):
        """Transform the data list to an array object and reshape it."""
        self.data = numx.array(self.data, dtype = self.dtype)
        self.data.shape = (self.tlen, self.input_dim)
        

### Extension Mechanism ###

# TODO: allow the functions that are registred to have an individual name,
#    provide the name as an additional argument 
# TODO: in the future could use ABC's to register nodes with extension nodes
# TODO: allow optional setup and restore methods that are called for a node
#    when the extension is activated. This could for example add special
#    attributes.


# dict of dicts of dicts, contains a key for each extension,
# the inner dict maps the node types to their extension node,
# the innermost dict then maps method names to functions
_extensions = dict()

# set containing the names of the currently activated extensions
_active_extensions = set()


class ExtensionException(mdp.MDPException):
    """Base class for extension related exceptions."""
    pass


def _register_function(ext_name, node_cls, func):
    """Register a function as an extension method.
    
    ext_name -- String with the name of the extension.
    node_cls -- Node class for which the method should be registered.
    func -- Function to be registered as an extension method.
    """
    method_name = func.__name__
    # perform safety check
    if method_name in node_cls.__dict__:
        original_method = getattr(node_cls, method_name)
        if not isinstance(original_method, types.MethodType):
            err = ("Extension method " + method_name + " tries to "
                   "override non-method attribute in class " +
                   str(node_cls))
            raise ExtensionException(err)
    _extensions[ext_name][node_cls][method_name] = func
    # do not set this now to be more flexibel
    func.__ext_original_method = None
    func.__ext_extension_name = ext_name

def extension_method(ext_name, node_cls):
    """Returns a function to register a function as extension method.
    
    This function is intendet to be used with the decorator syntax.
    
    ext_name -- String with the name of the extension.
    node_cls -- Node class for which the method should be registered.
    """
    def register_function(func):
        if not ext_name in _extensions:
            err = ("No ExtensionNode base class has been defined for this "
                   "extension.")
            raise ExtensionException(err)
        if not node_cls in _extensions[ext_name]:
            # register this node
            _extensions[ext_name][node_cls] = dict()
        _register_function(ext_name, node_cls, func)
        return func
    return register_function


class ExtensionNodeMetaclass(NodeMetaclass):
    """This is the metaclass for node extension superclasses.
    
    It takes care of registering extensions and the methods in the
    extension.
    """
    
    def __new__(cls, classname, bases, members):
        """Create new node classes and register extensions.
        
        If a concrete extension node is created then a corresponding mixin
        class is automatically created and registered.
        """
        if classname == "ExtensionNode":
            # initial creation of ExtensionNode class
            return super(ExtensionNodeMetaclass, ExtensionNodeMetaclass). \
                        __new__(cls, classname, bases, members)
        if ExtensionNode in bases:
            ext_name = members["extension_name"]
            if ext_name not in _extensions:
                # creation of a new extension, add entry in dict
                _extensions[ext_name] = dict()
        # find node that this extension is for
        base_node_cls = None
        for base in bases:
            if type(base) is not ExtensionNodeMetaclass:
                if base_node_cls is None:
                    base_node_cls = base
                else:
                    err = ("Extension node derived from multiple "
                           "normal nodes.")
                    raise ExtensionException(err)
        if base_node_cls is None:
            return super(ExtensionNodeMetaclass, ExtensionNodeMetaclass). \
                        __new__(cls, classname, bases, members)
        ext_node_cls = super(ExtensionNodeMetaclass, ExtensionNodeMetaclass). \
                        __new__(cls, classname, bases, members)
        ext_name = ext_node_cls.extension_name
        if not ext_name:
            err = "No extension name has been specified."
            raise ExtensionException(err)
        if not base_node_cls in _extensions[ext_name]:
            # register the base node
            _extensions[ext_name][base_node_cls] = dict()
        # register methods
        for member in members.values():
            if isinstance(member, types.FunctionType):
                _register_function(ext_name, base_node_cls, member)
        return ext_node_cls
                                                     

class ExtensionNode(object):
    """Base class for extensions nodes.
    
    A new extension node class should override the _extension_name.
    The concrete node implementations are then derived from this extension
    node class.
    
    Important note:
    To call a method from a parent class use:
        parent_class.method.im_func(self)
    """
    __metaclass__ = ExtensionNodeMetaclass
    # override this name in a concrete extension node base class
    extension_name = None


def get_extensions():
    """Return a dict with the currently registered extensions."""
    return _extensions

def get_active_extensions():
    """Returns the set with the names of the currently activated extensions."""
    # use copy to protect the original set, also important if the return
    # value is used in a for-loop (see deactivate_extensions function)
    return _active_extensions.copy()
    
def activate_extension(extension_name):
    """Activate the extension by injecting the extension methods."""
    if extension_name in _active_extensions:
        return
    _active_extensions.add(extension_name)
    try:
        for node_cls, methods in _extensions[extension_name].items():
            for method_name, method in methods.items():
                if method_name in node_cls.__dict__:
                    original_method = getattr(node_cls, method_name)
                    ## perform safety checks
                    # same check as in _register_function
                    if not isinstance(original_method, types.MethodType):
                        err = ("Extension method " + method_name + " tries to "
                               "override non-method attribute in class " +
                               str(node_cls))
                        raise ExtensionException(err)
                    if hasattr(original_method, "__ext_extension_name"):
                        err = ("Method name overlap for method '" + method_name +
                               "' between extension '" +
                               getattr(original_method, "__ext_extension_name") +
                               "' and newly activated extension '" +
                               extension_name + "'.")
                        raise ExtensionException(err)
                    method.__ext_original_method = original_method
                setattr(node_cls, method_name, method)
    except:
        # make sure that an incomplete activation is reverted
        deactivate_extension(extension_name)
        raise

def deactivate_extension(extension_name):
    """Deactivate the extension by removing the injected methods."""
    if extension_name not in _active_extensions:
        return
    for node_cls, methods in _extensions[extension_name].items():
        for method_name, method in methods.items():
            if method.__ext_original_method is not None:
                original_method = getattr(method, "__ext_original_method")
                setattr(node_cls, method_name, original_method)
                method.__ext_original_method = None
            else:
                # if the activation process failed then the extension method
                # might be mussing, so be tolerant
                try:
                    delattr(node_cls, method_name)
                except AttributeError:
                    pass
    _active_extensions.remove(extension_name)

def activate_extensions(extension_names):
    """Activate all the extensions for the given list of names."""
    try:
        for extension_name in extension_names:
            activate_extension(extension_name)
    except:
        # if something goes wrong deactivate all, otherwise we might be
        # in an inconsistent state (e.g. methods for active extensions might
        # have been removed)
        deactivate_extensions(get_active_extensions())
        raise

def deactivate_extensions(extension_names):
    """Deactivate all the extensions for the given list of names.
    
    extension_names -- Sequence of 
    """
    for extension_name in extension_names:
        deactivate_extension(extension_name)

# TODO: use the signature preserving decorator technique
def with_extension(extension_name):
    """Return a wrapper function to activate and deactivate the extension.
    
    This function is intended to be used with the decorator syntax.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                activate_extension(extension_name)
                result = func(*args, **kwargs)
            finally:
                deactivate_extension(extension_name)
            return result
        return wrapper
    return decorator
        
