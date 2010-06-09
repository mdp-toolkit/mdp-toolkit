"""
Coroutine decorator for BiNode methods.
"""

import mdp
import bimdp

# TODO: add option for default args in decorator
#    either by adding a new argument 'defaults', taking a dict
#    or by accepting a signature string, which is more convenient

# TODO: add short description in the tutorial, mentioned the DBN example
#    and the unittests as simple examples


class CoroutineBiNodeMixin(bimdp.BiNode):
    
    def __init__(self, **kwargs):
        super(CoroutineBiNodeMixin, self).__init__(**kwargs)
        # keys are the original method names
        self._coroutine_instances = {}
        
    def bi_reset(self):
        # delete the instance dict entries to unshadow the init methods
        for key in self._coroutine_instances:
            delattr(self, key)
        self._coroutine_instances = {}
        super(CoroutineBiNodeMixin, self).bi_reset()

   
def binode_coroutine(args, stop_message=False):
    """Decorator for the convenient definition of BiNode couroutines.
    
    Note that this decorator should be used with the CoroutineMixin to
    guarantee correct reseting in case of an exception.
    
    args -- List of string names of the arguments.
    stop_message -- Flag to signal if this coroutine is used during the
        stop_message phase. If this is False then the first value in 'args'
        must be 'x'. 
    
    Internally there are three methods/functions:
        - The user defined function containing the original coroutine code.
          This is only stored in the decorator closure.
        - A new method ('_coroutine_initialization') with the name and
          signature  of the decorated coroutine, which internally handles the
          first initialization of the coroutine instance.
          This method is returned by the decorator.
        - A method with the signature specified by the 'args' for the
          decorator.
          After the coroutine has been initialized this
          method shadows the initialization method in the class instance
          (using an instance attribute to shadow the class attribute).
    """
    if not stop_message and args[0] != "x":
        err = ("First argument must be 'x' unless 'stop_message' is set "
               "to True.")
        raise Exception(err)
    args = ["self"] + args
    def _binode_coroutine(coroutine):
        # the original coroutine is only stored in this closure
        infodict = mdp.NodeMetaclass._get_infodict(coroutine)
        original_name = infodict["name"]
        ## create the coroutine interface method        
        def _coroutine_interface(self, *args):
            try:
                return self._coroutine_instances[original_name].send(args)
            except StopIteration, exception:
                delattr(self, original_name)
                del self._coroutine_instances[original_name]
                if len(exception.args):
                    return exception.args
                else:
                    return None
        # turn the signature into the one specified by the args
        interface_infodict = infodict.copy()
        interface_infodict["argnames"] = args
        interface_infodict["defaults"] = ()
        interface_infodict["signature"] = ", ".join(args)
        coroutine_interface = mdp.NodeMetaclass._wrap_function(
                                    _coroutine_interface, interface_infodict)
        ## create the initialization method
        def _coroutine_initialization(self, *args):
            coroutine_instance = coroutine(self, *args)
            # better than using new.instancemethod
            bound_coroutine_interface = coroutine_interface.__get__(
                                                        self, self.__class__) 
            self._coroutine_instances[original_name] = coroutine_instance
            setattr(self, original_name, bound_coroutine_interface)
            return coroutine_instance.next()
        coroutine_initialization = mdp.NodeMetaclass._wrap_function(
                                    _coroutine_initialization, infodict)
        return coroutine_initialization
    return _binode_coroutine
        