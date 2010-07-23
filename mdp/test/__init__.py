import os
SCRIPT="run_tests.py"

def test(filename=None, keyword=None, seed=None):
    """Run MDP tests.

       filename -- only run tests in filename. If not set run all tests.
                   You do not need the full path, the relative path within the
                   test directory is enough.

       keyword  -- only run test items matching the given space separated
                   keywords.  precede a keyword with '-' to negate.
                   Terminate the expression with ':' to treat a match as
                   a signal to run all subsequent tests.

       seed     -- set random seed

    """
    if filename is None:
        loc = os.path.dirname(__file__)
    else:
        loc = os.path.join(os.path.dirname(__file__), os.path.basename(filename))
    args = []
    if keyword is not None:
        args.extend(('-k', str(keyword)))
    if seed is not None:
        args.extend(('--seed', str(seed)))

    args.append(loc)
    _worker = get_worker()
    return _worker(args)

def subtest(script, args):
        # run the auto-generated script in a subprocess
        import subprocess
        import sys
        subprocess.Popen([script]+args, stdout = sys.stdout, stderr = sys.stderr)

def get_worker():
    try:
        # use py.test module interface if it's installed
        import py
        return py.cmdline.pytest
    except ImportError:
        # try to locate the script
        script = os.path.join(os.path.dirname(__file__), SCRIPT)
        if os.path.exists(script):
            return lambda args: subtest(script, args)
        else:
            raise Exception('Could not find self-contained py.test script in'
                            '"%s"'%script)
