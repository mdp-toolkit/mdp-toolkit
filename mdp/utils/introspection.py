import inspect
import cPickle
import mdp

class _Walk(object):
    """Recursively crawl an object and search for attributes that
    are reference to Numeric arrays, return a dictionary:
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
        for name in dir(x):
            obj = getattr(x, name)
            if id(obj) in self.allobjs.keys():
                continue
            else:
                self.allobjs[id(obj)] = None
                
            if start:
                struct = '.'.join((start, name))
            else:
                struct = name
               
            if isinstance(obj, mdp.numx.ndarray):
                if start is not None:
                    arrays[struct] = obj
                else:
                    arrays[name] = obj
            elif name.startswith('__'):
                continue
            elif inspect.ismethod(obj):
                continue        
            elif isinstance(obj, object):
                arrays.update(self(obj, start = struct))
            else:
                raise NotImplementedError, 'object %s not known'%(str(obj))
        self.start = start
        return arrays

def _format_dig(dict):
    longest_name = max(map(len, dict.keys()))
    longest_size = max(map(lambda x: len('%d'%x[0]), dict.values()))
    msgs = []
    total_size = 0
    for name in sorted(dict.keys()):
        size = dict[name][0]
        total_size += size
        pname = (name+':').ljust(longest_name+1)
        psize = ('%d bytes'%size).rjust(longest_size+6)
        msg = "%s %s"%(pname, psize)
        msgs.append(msg)
    final = "Total %d arrays (%d bytes)"%(len(dict), total_size)
    msgs.append(final)
    return '\n'.join(msgs)
    
def dig_node(x):
    """Crawl recursively an MDP Node looking for arrays.
    Return (dictionary, string), where the dictionary is:
    { attribute_name: (size_in_bytes, array_reference)}
    and string is a nice string representation of it.
    """
    if not isinstance(x, mdp.Node):
        raise Exception, 'Cannot dig %s'%(str(type(x)))
    arrays = _Walk()(x)
    for name in arrays.keys():
        ar = arrays[name]
        if len(ar.shape) == 0:
            size = 1
        else:
            size = reduce(lambda x,y: x*y, ar.shape)
        bytes = ar.itemsize*size
        arrays[name] = (bytes, ar)
    return arrays, _format_dig(arrays)

def get_node_size(x):
    """Get node total byte-size using cPickle with protocol=2. (The byte-size
    has something to do with the memory needed by the node)."""
    size = len(cPickle.dumps(x, protocol = 2))
    return size

