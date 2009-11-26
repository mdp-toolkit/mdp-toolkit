"""
Extension Mechanism for nodes.

The extension mechanism makes it possible to dynamically add class attributes,
especially methods, for specific features to node classes
(e.g. for parallelization nodes need a _fork and _join method).
It is also possible for users to define new extensions to provide new
functionality for MDP nodes without having to modify any MDP code.

Without the extension mechanism extending nodes would be done by inheritance,
which is fine unless one wants to use multiple inheritance at the same time
(requiring multiple inheritance for every combination of extensions one wants
to use). The extension mechanism does not depend on inheritance, instead it
adds the methods to the node classes dynamically at runtime. This makes it
possible to activate extensions just when they are needed, reducing the risk
of interference between different extensions.  

However, since the extension mechanism provides a special Metaclass it is
still possible to define the extension nodes as classes derived from nodes.
This keeps the code readable and is compatible with automatic code checkers
(like the background pylint checks in the Eclipse IDE with PyDev).
"""

from mdp import MDPException, NodeMetaclass

# TODO: note the ParllelBiFlowNode purge_nodes method, which is not part
#    of the ParallelNode interface. Allow this?

# TODO: Register the node instances as well?
#    This would allow instance initialization when an extension is activated.
#    Implementing this should not be too hard via the metclass.

# TODO: Add warning about overriding public methods with respect to
#    the docstring wrappers?

# TODO: in the future could use ABC's to register nodes with extension nodes


# name prefix used for the original attributes when they are shadowed
ORIGINAL_ATTR_PREFIX = "_non_extension_"
# prefix used to store the current extension name for an attribute 
EXTENSION_ATTR_PREFIX = "_extension_for_"
# list of attribute names that are not affected by extensions,
NON_EXTENSION_ATTRIBUTES = ["__module__", "__doc__", "extension_name"]

# dict of dicts of dicts, contains a key for each extension,
# the inner dict maps the node types to their extension node,
# the innermost dict then maps attribute names to values
# (e.g. a method name to the actual function)
_extensions = dict()
# set containing the names of the currently activated extensions
_active_extensions = set()


class ExtensionException(MDPException):
    """Base class for extension related exceptions."""
    pass


def _register_attribute(ext_name, node_cls, attr_name, attr_value):
    """Register an attribute as an extension attribute.
    
    ext_name -- String with the name of the extension.
    node_cls -- Node class for which the method should be registered.
    """
    _extensions[ext_name][node_cls][attr_name] = attr_value
    
def extension_method(ext_name, node_cls, method_name=None):
    """Returns a function to register a function as extension method.
    
    This function is intended to be used with the decorator syntax.
    
    ext_name -- String with the name of the extension.
    node_cls -- Node class for which the method should be registered.
    method_name -- Name of the extension method (default value is None).
        If no value is provided then the name of the function is used.
        
    Note that it is possible to directly call other extension functions, call
    extension methods in other node classes or to use super in the normal way
    (the function will be called as a method of the node class).
    """
    def register_function(func):
        _method_name = method_name
        if not _method_name:
            _method_name = func.__name__
        if not ext_name in _extensions:
            err = ("No ExtensionNode base class has been defined for this "
                   "extension.")
            raise ExtensionException(err)
        if not node_cls in _extensions[ext_name]:
            # register this node
            _extensions[ext_name][node_cls] = dict()
        _register_attribute(ext_name, node_cls, _method_name, func)
        return func
    return register_function


class ExtensionNodeMetaclass(NodeMetaclass):
    """This is the metaclass for node extension superclasses.
    
    It takes care of registering extensions and the attributes in the
    extension.
    """
    
    def __new__(cls, classname, bases, members):
        """Create new node classes and register extensions.
        
        If a concrete extension node is created then a corresponding mixin
        class is automatically created and registered.
        """
        if classname == "ExtensionNode":
            # initial creation of ExtensionNode class
            return super(ExtensionNodeMetaclass, cls).__new__(cls, classname,
                                                              bases, members)
        # check if this is a new extension definition,
        # in that case this node is directly derived from ExtensionNode
        if ExtensionNode in bases:
            ext_name = members["extension_name"]
            if not ext_name:
                err = "No extension name has been specified."
                raise ExtensionException(err)
            if ext_name not in _extensions:
                # creation of a new extension, add entry in dict
                _extensions[ext_name] = dict()
            else:
                err = ("An extension with the name '" + ext_name +
                       "' has already been registered.")
                raise ExtensionException(err)
        # find the node that this extension node belongs to
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
            # this new extension is not directly derived from a node,
            # so there is nothing to register (no default implementation) 
            return super(ExtensionNodeMetaclass, cls).__new__(cls, classname,
                                                              bases, members)
        ext_node_cls = super(ExtensionNodeMetaclass, cls).__new__(
                                                cls, classname, bases, members)
        ext_name = ext_node_cls.extension_name
        if not base_node_cls in _extensions[ext_name]:
            # register the base node
            _extensions[ext_name][base_node_cls] = dict()
        # register methods
        for attr_name, attr_value in members.items():
            if attr_name not in NON_EXTENSION_ATTRIBUTES:
                _register_attribute(ext_name, base_node_cls,
                                    attr_name, attr_value)
        return ext_node_cls
                                                     

class ExtensionNode(object):
    """Base class for extensions nodes.
    
    A new extension node class should override the _extension_name.
    The concrete node implementations are then derived from this extension
    node class.
    
    To call an instance method from a parent class you have multiple options:
    
    - use super, but with the normal node class, e.g.:
        super(mdp.nodes.SFA2Node, self).method()
      Here SFA2Node was given instead of the extension node class for the
      SFA2Node.
      If the extensions node class is used directly (without the extension
      mechanism) this may lead to problems. In this case you have to be
      careful about the inheritance order and the effect on the MRO.
      
    - call it explicitly using the im_func attribute:
        parent_class.method.im_func(self)
        
    To call the original (pre-extension) method in the same class use you
    simply prefix the method name with '_non_extension_' (this is the value
    of the ORIGINAL_ATTR_PREFIX constant in this module).
    """
    __metaclass__ = ExtensionNodeMetaclass
    # override this name in a concrete extension node base class
    extension_name = None


def get_extensions():
    """Return a dictionary currently registered extensions.
    
    Note that this is not a copy, so if you change anything in this dict
    the whole extension mechanism will be affected. If you just want the
    names of the available extensions use get_extensions().keys().
    """
    return _extensions

def get_active_extensions():
    """Returns a list with the names of the currently activated extensions."""
    # use copy to protect the original set, also important if the return
    # value is used in a for-loop (see deactivate_extensions function)
    return list(_active_extensions)
    
def activate_extension(extension_name):
    """Activate the extension by injecting the extension methods."""
    if extension_name not in _extensions.keys():
        err = "Unknown extension name: %s" + str(extension_name)
        raise ExtensionException(err)
    if extension_name in _active_extensions:
        return
    _active_extensions.add(extension_name)
    try:
        for node_cls, attributes in _extensions[extension_name].items():
            for attr_name, attr_value in attributes.items():
                if attr_name in node_cls.__dict__:
                    if ((EXTENSION_ATTR_PREFIX + attr_name) in
                        node_cls.__dict__):
                        # two extensions override the same attribute
                        err = ("Name collision for attribute '" + 
                               attr_name + "' between extension '" +
                               getattr(node_cls,
                                       EXTENSION_ATTR_PREFIX + attr_name)
                               + "' and newly activated extension '" +
                               extension_name + "'.")
                        raise ExtensionException(err)
                    original_attr = getattr(node_cls, attr_name)
                    setattr(node_cls, ORIGINAL_ATTR_PREFIX + attr_name,
                            original_attr)
                setattr(node_cls, attr_name, attr_value)
                # store to which extension this attribute belongs, this is also
                # used as a flag that this is an extension attribute
                setattr(node_cls, EXTENSION_ATTR_PREFIX + attr_name,
                        extension_name)
    except:
        # make sure that an incomplete activation is reverted
        deactivate_extension(extension_name)
        raise

def deactivate_extension(extension_name):
    """Deactivate the extension by removing the injected methods."""
    if extension_name not in _extensions.keys():
        err = "Unknown extension name: " + str(extension_name)
        raise ExtensionException(err)
    if extension_name not in _active_extensions:
        return
    for node_cls, attributes in _extensions[extension_name].items():
        for attr_name in attributes.keys():
            original_name = ORIGINAL_ATTR_PREFIX + attr_name
            if original_name in node_cls.__dict__:
                # restore the original attribute
                original_attr = getattr(node_cls, original_name)
                setattr(node_cls, attr_name, original_attr)
                delattr(node_cls, original_name)
            else:
                try:
                    # no original attribute to restore, so simply delete
                    # might be missing if the activation failed
                    delattr(node_cls, attr_name)
                except AttributeError:
                    pass
            try:
                # might be missing if the activation failed
                delattr(node_cls, EXTENSION_ATTR_PREFIX + attr_name)
            except AttributeError:
                pass
    _active_extensions.remove(extension_name)

def activate_extensions(extension_names):
    """Activate all the extensions for the given names.
    
    extension_names -- Sequence of extension names.
    """
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
    """Deactivate all the extensions for the given names.
    
    extension_names -- Sequence of extension names.
    """
    for extension_name in extension_names:
        deactivate_extension(extension_name)

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
        # now make sure that docstring and signature match the original
        func_info = NodeMetaclass._get_infodict(func)
        return NodeMetaclass._wrap_function(wrapper, func_info)
    return decorator
        
