import mdp
import sys as _sys
import os as _os
import traceback as _traceback
import cPickle as _cPickle
import warnings as _warnings
import tempfile as _tempfile
import types as _types

# import numeric module (scipy, Numeric or numarray)
numx = mdp.numx

class CrashRecoveryException(mdp.MDPException):
    """Class to handle crash recovery """
    def __init__(self, *args):
        """Allow crash recovery.
        Arguments: (error_string, crashing_obj, parent_exception)
        The crashing object is kept in self.crashing_obj
        The triggering parent exception is kept in self.parent_exception.
        """
        errstr = args[0]
        self.crashing_obj = args[1]
        self.parent_exception = args[2]
        # ?? python 2.5: super(CrashRecoveryException, self).__init__(errstr)
        mdp.MDPException.__init__(self, errstr)

    def dump(self, filename = None):
        """
        Save a pickle dump of the crashing object on filename.
        If filename is None, the crash dump is saved on a file created by
        the tempfile module.
        Return the filename.
        """
        if filename is None:
            (fd, filename)=_tempfile.mkstemp(suffix=".pic",prefix="MDPcrash_")
            fl = _os.fdopen(fd, 'w+b', -1)
        else:
            fl = file(filename, 'w+b',-1)
        _cPickle.dump(self.crashing_obj,fl)
        fl.close()
        return filename

class FlowException(mdp.MDPException):
    """Base class for exceptions in Flow subclasses."""
    pass

class FlowExceptionCR(CrashRecoveryException, FlowException):
    """Class to handle flow-crash recovery """
    
    def __init__(self, *args):
        """Allow crash recovery.
        
        Arguments: (error_string, flow_instance, parent_exception)
        The triggering parent exception is kept in self.parent_exception.
        If flow_instance._crash_recovery is set, save a crash dump of
        flow_instance on the file self.filename"""
        CrashRecoveryException.__init__(self,*args)
        rec = self.crashing_obj._crash_recovery 
        errstr = args[0]
        if rec:
            if isinstance(rec, str):
                name = rec
            else:
                name = None
            name = CrashRecoveryException.dump(self,name)
            dumpinfo = '\nA crash dump is available on: "'+name+'"'
            self.filename = name
            errstr = errstr+dumpinfo    

        Exception.__init__(self, errstr)

class Flow(object):
    """A Flow consists in a linear sequence of Nodes.

    The data is sent to an input node and is successively processed
    by the following nodes on the graph. The Flow class
    automatizes training, execution and inverse execution
    (if defined) of the whole nodes sequence.
    Training can be supervised and can consist of multiple phases.
    
    Crash recovery is optionally available: in case of failure the
    current state of the flow is saved for later inspection.

    This class is a Python container class. Most of the builtin 'list'
    methods are available."""

    def __init__(self, flow, crash_recovery=False, verbose=False):
        """
        'flow' is a list of Nodes.
        The value of 'crash_recovery' is sent to 'set_crash_recovery'.
        If 'verbose' is set print some basic progress information."""
        self._check_nodes_consistency(flow)
        self.flow = flow
        self.verbose = verbose
        self.set_crash_recovery(crash_recovery)

    def _propagate_exception(self, except_, nodenr):
        # capture exception. the traceback of the error is printed and a
        # new exception, containing the identity of the node in the flow
        # is raised. Allow crash recovery.
        (type, val, tb) = _sys.exc_info()
        prev = ''.join(_traceback.format_exception(except_.__class__,
                                                  except_,tb))
        act = "\n! Exception in node #%d (%s):\n" \
              % (nodenr, str(self.flow[nodenr]))
        errstr =''.join(('\n', 40*'-', act, 'Node Traceback:\n', prev, 40*'-'))
        raise FlowExceptionCR,(errstr, self, except_)

    def _train_node(self, data_iterator, nodenr):
        #trains a single node in the flow
        node = self.flow[nodenr]
        if data_iterator is not None and not node.is_trainable():
            # attempted to train a node although it is not trainable.
            # raise a warning and continue with the next node.
            #wrnstr = "\n! Node %d is not trainable" % nodenr + \
            #         "\nYou probably need a 'None' iterator for"+\
            #         " this node. Continuing anyway."
            #_warnings.warn(wrnstr, mdp.MDPWarning)
            return
        elif data_iterator is None and node.is_training():
            # A None iterator is passed to a training node
            errstr = "\n! Node %d is training" % nodenr + \
                     " but received a 'None' iterator."
            raise FlowException, errstr
        elif data_iterator is None and not node.is_trainable():
            # skip training if node is not trainable
            return
            
        try:
            # We leave the last training phase open for the
            # CheckpointFlow class.
            # Checkpoint functions must close it explicitly if needed!
            # Note that the last training_phase is closed
            # automatically when the node is executed.
            while True:
                for x in data_iterator:
                    # the arguments following the first are passed only to the
                    # currently trained node, allowing the implementation of
                    # supervised nodes
                    if (type(x) is tuple) or (type(x) is list):
                        arg = x[1:]
                        x = x[0]
                    else:
                        arg = ()
                    # filter x through the previous nodes
                    if nodenr > 0: x = self._execute_seq(x, nodenr-1)
                    # train current node
                    node.train(x, *arg)
                if node.get_remaining_train_phase() > 1:
                    # close the previous training phase
                    node.stop_training()
                else:
                    break
        except mdp.TrainingFinishedException, e:
            # attempted to train a node although its training phase is already
            # finished. raise a warning and continue with the next node.
            wrnstr = "\n! Node %d training phase already finished" % nodenr +\
                     " Continuing anyway."
            _warnings.warn(wrnstr, mdp.MDPWarning)
        except FlowExceptionCR, e:
            # this exception was already propagated,
            # probably during the execution  of a node upstream in the flow
            (exc_type, val) = _sys.exc_info()[:2]
            prev = ''.join(_traceback.format_exception_only(e.__class__,e))
            prev = prev[prev.find('\n')+1:]
            act = "\nWhile training node #%d (%s):\n" \
              % (nodenr, str(self.flow[nodenr]))
            errstr = ''.join(('\n', 40*'=', act, prev, 40*'='))
            raise FlowException(errstr)
        except Exception, e:
            # capture any other exception occured during training.
            self._propagate_exception(e, nodenr)
            
    def _train_check_iterators(self, data_iterators):
        #verifies that the number of iterators matches that of
        #the signal nodes and multiplies them if needed.
        flow = self.flow

        if isinstance(data_iterators, numx.ndarray):
            data_iterators = [[data_iterators]]*len(flow)

        if not isinstance(data_iterators, list):
            errstr = "'data_iterators' is "+ str(type(data_iterators)) + \
                     " must be either a list of iterators or an array"
            raise FlowException, errstr

        # check that all elements are iterable
        for i in range(len(data_iterators)):
            el = data_iterators[i]
            if el is not None and not hasattr(el, '__iter__'):
                raise FlowException, "Element number %d in the " % i + \
                      "iterators list is not a list or iterator."
       
        # check that the number of data_iterators is correct
        if len(data_iterators)!=len(flow):
            error_str = "%d data iterators specified, %d needed"
            raise FlowException, error_str % (len(data_iterators), len(flow))

        # check that every node with multiple phases has an iterator
        # but NOT a generator (since you cannot "rewind" a generator)
        for i in range(len(flow)):
            node, iter = flow[i], data_iterators[i]
            if len(node._train_seq)>1 and type(iter) is _types.GeneratorType:
                errstr = "Node number %d has multiple training phases " %i +\
                         "but the corresponding iterator is a generator. " +\
                         "This is not allowed since generators cannot be " +\
                         "'rewinded' for further training."
                raise FlowException, errstr

        return data_iterators

    def _close_last_node(self):
        if self.verbose: print "Close the training phase of the last node"
        try:
            self.flow[-1].stop_training()
        except mdp.TrainingFinishedException:
            pass
        except Exception, e:
            self._propagate_exception(e, len(self.flow)-1)

    def set_crash_recovery(self, state = True):
        """Set crash recovery capabilities.
        
        When a node raises an Exception during training, execution, or
        inverse execution that the flow is unable to handle, a FlowExceptionCR
        is raised. If crash recovery is set, a crash dump of the flow
        instance is saved for later inspection. The original exception
        can be found as the 'parent_exception' attribute of the
        FlowExceptionCR instance.
        
        - If 'state' = False, disable crash recovery.
        - If 'state' is a string, the crash dump is saved on a file
          with that name.
        - If 'state' = True, the crash dump is saved on a file created by
          the tempfile module.
        """
        self._crash_recovery = state

    def train(self, data_iterators):
        """Train all trainable nodes in the flow.
        
        'data_iterators' is a list of iterators (note that a list is also
        an iterator), which return data arrays, one for each node in the flow.
        If instead one array is specified, it is used as input training
        sequence for all nodes.
        
        Instead of a data array x the iterators can also return a list or tuple, 
        where the first entry is x and the following are args for the training 
        of the node (e.g. for supervised training). 

        Generator-type iterators are supported only for nodes with
        a single training phase (this is because they cannot be
        restarted after they expired).
        """

        data_iterators = self._train_check_iterators(data_iterators)
        
        # train each Node successively
        for i in range(len(self.flow)):
            if self.verbose: print "Training node #%d (%s)" \
               % (i,str(self.flow[i]))
            self._train_node(data_iterators[i], i)
            if self.verbose: print "Training finished"

        self._close_last_node()

    def _execute_seq(self, x, nodenr = None):
        # Filters input data 'x' through the nodes 0..'node_nr' included
        flow = self.flow
        if nodenr is None: nodenr = len(flow)-1
        for i in range(nodenr+1):
            try:
                x = flow[i].execute(x)
            except Exception, e:
                self._propagate_exception(e, i)
        return x

    def execute(self, iterator, nodenr = None):
        """Process the data through all nodes in the flow.
        
        'iterator' is an iterator (note that a list is also an iterator),
        which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.
        
        If 'nodenr' is specified, the flow is executed only up to
        node nr. 'nodenr'.
        This is equivalent to 'flow[:nodenr+1](iterator)'."""
        # if iterator is one single input sequence
        if isinstance(iterator, numx.ndarray):
            return self._execute_seq(iterator, nodenr)
        # otherwise it is a iterator
        res = []
        for x in iterator:
            res.append(self._execute_seq(x, nodenr))
        return numx.concatenate(res)

    def _inverse_seq(self, x):
        #Successively invert input data 'x' through all nodes backwards
        flow = self.flow
        for i in range(len(flow)-1,-1,-1):
            try:
                x = flow[i].inverse(x)
            except Exception, e:
                self._propagate_exception(e, i)
        return x


    def inverse(self, iterator):
        """Process the data through all nodes in the flow backwards        
        (starting from the last node up to the first node) by calling the
        inverse function of each node. Of course, all nodes in the
        flow must be invertible.
        
        'iterator' is an iterator  (note that a list is also an iterator),
        which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.
        
        Note that this is _not_ equivalent to 'flow[::-1](iterator)',
        which also executes the flow backwards but calls the 'execute'
        function of each node."""
        
        # if iterator is one single input sequence
        if isinstance(iterator, numx.ndarray):
            return self._inverse_seq(iterator)
        # otherwise it is a iterator
        res = []
        for x in iterator:
            res.append(self._inverse_seq(x))
        return numx.concatenate(res)

    def copy(self, protocol = -1):
        """Return a deep copy of the flow.
        Protocol is the pickle protocol."""
        as_str = _cPickle.dumps(self, protocol)
        return _cPickle.loads(as_str)

    def save(self, filename, protocol = -1):
        """Save a pickled representation of the flow to 'filename'.
        If 'filename' is None, return a string.
        
        Note: the pickled Flow is not guaranteed to be upward or
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

    def __call__(self, iterator, nodenr = None):
        """Calling an instance is equivalent to call its 'execute' method."""
        return self.execute(iterator, nodenr=nodenr)

    ###### string representation
    def __str__(self):
        nodes = ', '.join([str(x) for x in self.flow])
        return '['+nodes+']'

    def __repr__(self):
        # this should look like a valid Python expression that
        # could be used to recreate an object with the same value
        # eval(repr(object)) == object
        name = type(self).__name__
        pad = len(name)+2
        sep = ',\n'+' '*pad
        nodes = sep.join([repr(x) for x in self.flow])
        return '%s(%s)' %(name, '['+nodes+']')

    ###### private container methods

    def __len__(self):
        return len(self.flow)

    def _check_dimension_consistency(self, out, inp):
        """Raise ValueError when both dimensions are set and different."""
        if ((out and inp) is not None) and out != inp:
            errstr = "dimensions mismatch: %d != %d" % (out, inp)
            raise ValueError, errstr

    def _check_nodes_consistency(self, flow = None):
        """Check the dimension consistency of a list of nodes."""
        if flow is None:
            flow = self.flow
        len_flow = len(flow)
        for i in range(1,len_flow):
            out = flow[i-1].output_dim
            inp = flow[i].input_dim
            self._check_dimension_consistency(out, inp)

    def _check_value_type_isnode(self, value):
        if not isinstance(value, mdp.Node):
            raise TypeError, "flow item must be Node instance"

    def __getitem__(self, key):
        if isinstance(key, slice):
            flow_slice = self.flow[key]
            self._check_nodes_consistency(flow_slice)
            return self.__class__(flow_slice)
        else:
            return self.flow[key]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            map(self._check_value_type_isnode, value)
        else:
            self._check_value_type_isnode(value)

        # make a copy of list
        flow_copy = list(self.flow)
        flow_copy[key] = value
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy

    def __delitem__(self, key):
        # make a copy of list
        flow_copy = list(self.flow)
        del flow_copy[key]
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy

    def __contains__(self, item):
        return self.flow.__contains__(item)
    
    def __iter__(self):
        return self.flow.__iter__()
    
    def __add__(self, other):
        # append other to self
        if isinstance(other, Flow):
            flow_copy = list(self.flow).__add__(other.flow)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, mdp.Node):
            flow_copy = self.copy()
            flow_copy.append(other)
            return flow_copy
        else:
            err_str = 'can only concatenate flow'+ \
                      ' (not \'%s\') to flow'%(type(other).__name__) 
            raise TypeError, err_str

    ###### public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]

    def extend(self, x):
        """flow.extend(iterable) -- extend flow by appending
        elements from the iterable"""
        if not isinstance(x, Flow):
            err_str = 'can only concatenate flow'+ \
                      ' (not \'%s\') to flow'%(type(x).__name__) 
            raise TypeError, err_str
        self[len(self):len(self)] = x

    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]

    def pop(self, i = -1):
        """flow.pop([index]) -> node -- remove and return node at index
        (default last)"""
        x = self[i]
        del self[i]
        return x

class CheckpointFlow(Flow):
    """Subclass of Flow class that allows user-supplied checkpoint functions
    to be executed at the end of each phase, for example to
    save the internal structures of a node for later analysis."""
    
    def _train_check_checkpoints(self, checkpoints):
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]*len(self.flow)
        
        if len(checkpoints) != len(self.flow):
            error_str = "%d checkpoints specified, %d needed"
            raise FlowException, error_str % (len(checkpoints), len(self.flow))

        return checkpoints


    def train(self, data_iterators, checkpoints):
        """Train all trainable nodes in the flow.

        Additionally calls the checkpoint function 'checkpoint[i]'
        when the training phase of node #i is over.
        A checkpoint function takes as its only argument the trained node.
        If the checkpoint function returns a dictionary, its content is
        added to the instance's dictionary.
        The class CheckpointFunction can be used to define user-supplied
        checkpoint functions"""

        data_iterators = self._train_check_iterators(data_iterators)
        checkpoints = self._train_check_checkpoints(checkpoints)

        # train each Node successively
        for i in range(len(self.flow)):
            node = self.flow[i]
            if self.verbose:
                print "Training node #%d (%s)" % (i,type(node).__name__)
            self._train_node(data_iterators[i], i)
            if (i <= len(checkpoints)) and (checkpoints[i] is not None):
                dict = checkpoints[i](node)
                if dict: self.__dict__.update(dict)
            if self.verbose: print "Training finished"

        self._close_last_node()

class CheckpointFunction(object):
    """Base class for checkpoint functions.    
    This class can be subclassed to build objects to be used as a checkpoint
    function in a CheckpointFlow. Such objects would allow to define parameters
    for the function and save informations for later use."""

    def __call__(self, node):
        """Execute the checkpoint function.

        This is the method that is going to be called at the checkpoint.
        Overwrite it to match your needs."""
        pass

class CheckpointSaveFunction(CheckpointFunction):
    """This checkpoint function saves the node in pickle format.
    The pickle dump can be done either before the training phase is finished or
    right after that.
    In this way, it is for example possible to reload it in successive sessions
    and continue the training.
    """

    def __init__(self, filename, stop_training = 0, binary = 1, protocol = 2):
        """CheckpointSaveFunction constructor.
        
        'filename'      -- the name of the pickle dump file.
        'stop_training' -- if set to 0 the pickle dump is done before
                           closing the training phase
                           if set to 1 the training phase is closed and then
                           the node is dumped
        'binary'        -- sets binary mode for opening the file.
                           When using a protocol higher than 0, make sure
                           the file is opened in binary mode. 
        'protocol'      -- is the 'protocol' argument for the pickle dump
                           (see Pickle documentation for details)
        """
        self.filename = filename
        self.proto = protocol
        self.stop_training = stop_training
        if binary or protocol > 0:
            self.mode = 'wb'
        else:
            self.mode = 'w'

    def __call__(self, node):
        fid = open(self.filename, self.mode)
        if self.stop_training: node.stop_training()
        _cPickle.dump(node, fid, self.proto)
        fid.close()
