from builtins import str
import os
from mdp.configuration import _version_too_old

def test(filename=None, seed=None, options='', mod_loc=None):
    """Run tests.

       filename -- only run tests in filename. If not set run all tests.
                   You do not need the full path, the relative path within the
                   test directory is enough.

       seed     -- set random seed

       options  -- options to be passed to py.test (as a string)

       mod_loc  -- don't use it, it's for internal usage
    """
    if mod_loc is None:
        mod_loc = os.path.dirname(__file__)
    if filename is None:
        loc = mod_loc
    else:
        loc = os.path.join(mod_loc, os.path.basename(filename))
    args = []
    if seed is not None:
        args.append('--seed=' + str(seed))
    # add --assert=reiterp option to work around permissions problem
    # with __pycache__ directory when MDP is installed on a normal
    # user non-writable directory
    options = "--assert=reinterp "+options
    args.extend(options.split())
    args.append(loc)
    _worker = get_worker()
    return _worker(args)

def get_worker():
    try:
        import py.test
    except ImportError:
        raise ImportError('You need py.test to run the test suite!')

    # check that we have at least version 2.1.2
    if _version_too_old(py.test.__version__, (2,1,2)):
        raise ImportError('You need at least py.test version 2.1.2,'
                ' found %s!'%py.test.__version__)
    else:
        return py.test.cmdline.main
