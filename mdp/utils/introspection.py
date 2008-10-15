import types
import cPickle
import mdp

class _Walk(object):
    """Recursively crawl an object and search for attributes that
    are reference to numpy arrays, return a dictionary:
    {attribute_name: array_reference}.

    Usage:
        _Walk()(object)
    """
    def __init__(self):
        self.arrays = {}
        self.start = None
        self.allobjs = {}
        
    def __call__(self, x, start = None):
        arrays = self.arrays
        # loop through the object dictionary
        for name in dir(x):
            # get the corresponding member
            obj = getattr(x, name)
            if id(obj) in self.allobjs.keys():
                # if we already examined the member, skip to the next
                continue
            else:
                # add the id of this object to the list of know members
                self.allobjs[id(obj)] = None

            if start is None:
                # initialize a string structure to keep track of array names
                struct = name
            else:
                # struct is x.y.z (where x and y are objects and z an array)
                struct = '.'.join((start, name))

            if isinstance(obj, mdp.numx.ndarray):
                # the present member is an array
                # add it to the dictionary of all arrays
                if start is not None:
                    arrays[struct] = obj
                else:
                    arrays[name] = obj
            elif name.startswith('__') or type(obj) in (int, long, float,
                                                        types.MethodType):
                # the present member is a private member or a known
                # type that does not support arrays as attributes
                # Note: this is to avoid infinite
                # recursion in python2.6. Just remove the "or type in ..."
                # condition to see the error. There must be a better way.
                continue
            else:
                # we need to examine the present member in more detail
                arrays.update(self(obj, start = struct))
        self.start = start
        return arrays

def _format_dig(dict_):
    longest_name = max(map(len, dict_.keys()))
    longest_size = max(map(lambda x: len('%d'%x[0]), dict_.values()))
    msgs = []
    total_size = 0
    for name in sorted(dict_.keys()):
        size = dict_[name][0]
        total_size += size
        pname = (name+':').ljust(longest_name+1)
        psize = ('%d bytes' % size).rjust(longest_size+6)
        msg = "%s %s" % (pname, psize)
        msgs.append(msg)
    final = "Total %d arrays (%d bytes)" % (len(dict_), total_size)
    msgs.append(final)
    return '\n'.join(msgs)
    
def dig_node(x):
    """Crawl recursively an MDP Node looking for arrays.
    Return (dictionary, string), where the dictionary is:
    { attribute_name: (size_in_bytes, array_reference)}
    and string is a nice string representation of it.
    """
    if not isinstance(x, mdp.Node):
        raise Exception('Cannot dig %s' % (str(type(x))))
    arrays = _Walk()(x)
    for name in arrays.keys():
        ar = arrays[name]
        if len(ar.shape) == 0:
            size = 1
        else:
            size = reduce(lambda x, y: x*y, ar.shape)
        bytes = ar.itemsize*size
        arrays[name] = (bytes, ar)
    return arrays, _format_dig(arrays)

def get_node_size(x):
    """Get node total byte-size using cPickle with protocol=2. (The byte-size
    is related to the memory needed by the node)."""
    size = len(cPickle.dumps(x, protocol = 2))
    return size

