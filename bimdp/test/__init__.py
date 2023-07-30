import os
import mdp
from mdp.test import test as mdp_test

# wrap the mdp.test.test function and set the module path to bimdp path
infodict = mdp.NodeMetaclass._function_infodict(mdp_test)
idx = infodict["argnames"].index('mod_loc')
defaults = list(infodict['defaults'])
defaults[idx] = os.path.dirname(__file__)
infodict['defaults'] = tuple(defaults)

test = mdp.NodeMetaclass._wrap_function(mdp_test, infodict)
