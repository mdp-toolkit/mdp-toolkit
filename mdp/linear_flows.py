import mdp
import sys
import traceback
import types
import cPickle
import warnings
import string
import tempfile
import os

# import numeric module (scipy, Numeric or numarray)
numx = mdp.numx

class FlowException(mdp.MDPException):
    """Base class for exceptions in SimpleFlow subclasses."""
    pass

class FlowExceptionCR(mdp.utils.CrashRecoveryException, FlowException):
    """Class to handle flow-crash recovery """
    def __init__(self, *args):
        """Allow crash recovery.
        
        Arguments: (error_string, flow_instance, parent_exception)
        The triggering parent exception is kept in self.parent_exception.
        If flow_instance._crash_recovery is set, save a crash dump of
        flow_instance on the file self.filename"""
        mdp.utils.CrashRecoveryException.__init__(self,*args)
        rec = self.crashing_obj._crash_recovery 
        errstr = args[0]
        if rec:
            if type(rec) is types.StringType:
                name = rec
            else:
                name = None
            name = mdp.utils.CrashRecoveryException.dump(self,name)
            dumpinfo = '\nA crash dump is available on: "'+name+'"'
            self.filename = name
            errstr = errstr+dumpinfo    

        Exception.__init__(self,errstr)


# use this if there's no need for a generator in the flow
void_generator = []

class SimpleFlow(object):
    """A SimpleFlow consists in a linear sequence of SignalNodes.

    The data is sent to an input node and is successively processed
    by the following nodes on the graph. The SimpleFlow class
    automatizes training, execution and inverse execution
    (if defined) of the whole nodes sequence.
    This class is a Python container class. Most of the builtin 'list'
    methods are available."""

    def __init__(self, flow, verbose = 0):
        """
        'flow' is a list of SignalNodes.
        If 'verbose' is set print some basic progress information."""
        self.flow = flow
        self.verbose = verbose
        self._crash_recovery = 0
        
    #?? add_signalnode, remove_signalnode could be more complicated because
    #   it needs to check for the compatibility of the dimensions

    def _propagate_exception(self, except_, nodenr):
        # capture exception. the traceback of the error is printed and a
        # new exception, containing the identity of the node in the flow
        # is raised. Allow crash recovery.
        tb = sys.exc_traceback
        prev = string.join(traceback.format_exception(except_.__class__,
                                                      except_,tb),'')
        act = "\n! Exception in node #%d (%s):\n" \
              % (nodenr, type(self.flow[nodenr]).__name__)
        errstr = '\n'+40*'-'+act+'Node Traceback:\n'+prev+40*'-'
        raise FlowExceptionCR,(errstr, self, except_)

    def _train_node(self, generator, nodenr):
        #trains a single node in the flow
        node = self.flow[nodenr]
        try:
            for x in generator:
                # filter x through the previous nodes
                if nodenr>0: x = self._execute_seq(x, nodenr-1)
                # train current node
                node.train(x)
        except mdp.IsNotTrainableException, e:
            # attempted to train a node although it is not trainable.
            # raise a warning and continue with the next node.
            wrnstr = "\n! Node %d in not trainable" % nodenr + \
                     "\nYou probably need a 'None' generator for"+\
                     " this node. Continuing anyway."
            warnings.warn(wrnstr, mdp.MDPWarning)
        except mdp.TrainingFinishedException, e:
            # attempted to train a node although its training phase is already
            # finished. raise a warning and continue with the next node.
            wrnstr = "\n! Node %d training phase already finished" % nodenr +\
                     " Continuing anyway."
            warnings.warn(wrnstr, mdp.MDPWarning)
        except Exception, e:
            # capture any other exception occured during training.
            self._propagate_exception(e, nodenr)
            
    def _train_check_generators(self, data_generators):
        #verifies that the number of generators matches that of
        #the signal node sand multiplies them if needed.
        flow = self.flow

        if type(data_generators) is numx.ArrayType:
            data_generators = [[data_generators]]*len(flow)
            
        if type(data_generators) is not types.ListType:
            errstr = "'data_generators' is "+ str(type(data_generators)) + \
                     " must be either a list of generators or an array"
            raise FlowException, errstr

        # check that all elements are generators or list
        for i in range(len(data_generators)):
            el = data_generators[i]
            if type(el) not in [types.ListType, types.GeneratorType]:
                if el==None:
                    data_generators[i] = []
                else:
                    raise FlowException, "Element number %d in the " % i + \
                          "generator list is not a list or generator."
        
        # check that the number of data_generators is correct
        if len(data_generators)!=len(flow):
            error_str = "%d data sequence generators specified, %d needed"
            raise FlowException, error_str % (len(data_generators), len(flow))

        return data_generators

    def _close_last_node(self):
        if self.verbose: print "Close the training phase of the last node"
        try:
            self.flow[-1].stop_training()
        except mdp.TrainingFinishedException:
            pass
        except Exception, e:
            self._propagate_exception(e, len(self.flow)-1)

    def set_crash_recovery(self, state = 1):
        """Set crash recovery capabilities.
        
        When a node raises an Exception during training, execution, or
        inverse execution that the flow is unable to handle, a FlowExceptionCR
        is raised. If crash recovery is set, a crash dump of the flow
        instance is saved for later inspection. The original exception
        can be found as the 'parent_exception' attribute of the
        FlowExceptionCR instance.
        
        - If 'state' = 0, disable crash recovery.
        - If 'state' is a string, the crash dump is saved on a file
          with that name.
        - If 'state' = 1, the crash dump is saved on a file created by
          the tempfile module.
        """
        self._crash_recovery = state

    def train(self, data_generators):
        """Train all trainable nodes in the flow.
        
        'data_generators' is a list of generators (note that a list is also
        a generator), which return data arrays, one for each node in the flow.
        If instead one array is specified, it is used as input training
        sequence for all nodes."""

        data_generators = self._train_check_generators(data_generators)
        
        # train each SignalNode successively
        for i in range(len(self.flow)):
            if self.verbose: print "Training node #%d (%s)" \
               % (i,str(self.flow[i]))
            self._train_node(data_generators[i], i)
            if self.verbose: print "Training finished"

        self._close_last_node()

    def _execute_seq(self, x, nodenr = None):
        # Filters input data 'x' through the nodes 0..'node_nr' included
        flow = self.flow
        if nodenr==None: nodenr = len(flow)-1
        for i in range(nodenr+1):
            try:
                x = flow[i].execute(x)
            except Exception, e:
                self._propagate_exception(e, i)
        return x

    def execute(self, generator, nodenr = None):
        """Process the data through all nodes in the flow.
        
        'generator' is a generator  (note that a list is also a generator),
        which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.
        
        If 'nodenr' is specified, the flow is executed only up to
        node nr. 'nodenr'.
        This is equivalent to 'flow[:nodenr+1](generator)'."""
        # if generator is one single input sequence
        if type(generator) is numx.ArrayType:
            return self._execute_seq(generator, nodenr)
        # otherwise it is a generator
        res = []
        for x in generator:
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


    def inverse(self, generator):
        """Process the data through all nodes in the flow backwards        
        (starting from the last node up to the first node) by calling the
        inverse function of each node.
        'generator' is a generator  (note that a list is also a generator),
        which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.
        
        Note that this is _not_ equivalent to 'flow[::-1](generator)',
        which also executes the flow backwards but calls the 'execute'
        function of each node."""
        
        # if generator is one single input sequence
        if type(generator) is numx.ArrayType:
            return self._inverse_seq(generator)
        # otherwise it is a generator
        res = []
        for x in generator:
            res.append(self._inverse_seq(x))
        return numx.concatenate(res)

    def copy(self, protocol = -1):
        """Return a deep copy of the flow.
        Protocol is the pickle protocol."""
        as_str = cPickle.dumps(self, protocol)
        return cPickle.loads(as_str)

    def __call__(self, generator, nodenr = None):
        """Calling an instance is equivalent to call its 'execute' method."""
        return self.execute(generator, nodenr=nodenr)

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

    def _check_nodes_consistency(self, flow):
        """Check the dimension consistency of a list of nodes."""
        len_flow = len(flow)
        for i in range(1,len_flow):
            out = flow[i-1].get_output_dim()
            inp = flow[i].get_input_dim()
            self._check_dimension_consistency(out, inp)

    def _check_value_type_isnode(self, value):
        if not isinstance(value, mdp.SignalNode):
            raise TypeError, "flow item must be SignalNode instance"

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
        if not isinstance(other, SimpleFlow):
            err_str = 'can only concatenate flow'+ \
                      ' (not \'%s\') to flow'%(type(other).__name__) 
            raise TypeError, err_str
        flow_copy = list(self.flow).__add__(other.flow)
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        # if no exception was raised, accept the new sequence
        return self.__class__(flow_copy)

    ###### public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]

    def extend(self, x):
        """flow.extend(iterable) -- extend flow by appending
        elements from the iterable"""
        if not isinstance(x, SimpleFlow):
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

        
class CheckpointFlow(SimpleFlow):
    """Subclass of SimpleFlow class allows user-supplied checkpoint functions
    to be executed at the end of each phase, for example to
    save the internal structures of a node for later analysis."""
    
    def _train_check_checkpoints(self, checkpoints):
        if type(checkpoints) is not types.ListType:
            checkpoints = [checkpoints]*len(self.flow)
        
        if len(checkpoints) != len(self.flow):
            error_str = "%d checkpoints specified, %d needed"
            raise FlowException, error_str % (len(checkpoints), len(self.flow))

        return checkpoints


    def train(self, data_generators, checkpoints):
        """Train all trainable nodes in the flow.

        Additionally calls the checkpoint function 'checkpoint[i]'
        when the training phase of node #i is over.
        A checkpoint function takes as its only argument the trained node.
        If the checkpoint function returns a dictionary, its content is
        added to the instance's dictionary.
        The class CheckpointFunction can be used to define user-supplied
        checkpoint functions"""

        data_generators = self._train_check_generators(data_generators)
        checkpoints = self._train_check_checkpoints(checkpoints)

        # train each SignalNode successively
        for i in range(len(self.flow)):
            node = self.flow[i]
            if self.verbose:
                print "Training node #%d (%s)" % (i,type(node).__name__)
            self._train_node(data_generators[i], i)
            if i<=len(checkpoints) and checkpoints[i]!=None:
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
    and prolongate the training.
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
        cPickle.dump(node, fid, self.proto)
        fid.close()
