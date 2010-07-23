collect_ignore = ['build', 'cover', 'html',
                  'mdp/parallel/pp_test',
                  ]

# global hooks for py.test
import mdp

_err_str = """
IMPORTANT: some tests use random numbers. This could
occasionally lead to failures due to numerical degeneracies.
To rule this out, please run the tests more than once.
If you get reproducible failures please report a bug!
"""

def pytest_addoption(parser):
    """Add random seed option to py.test"""
    parser.addoption("--seed", dest="seed", type="int", action="store",
                     help="set random seed")

def pytest_report_header(config):
    # TODO: insert here information from the new mdp.info object
    return "%s\nRandom Seed: %d" % (mdp.config.info(), config.option.seed)

def pytest_terminal_summary(terminalreporter):
    # TODO: insert here information from the new mdp.info object
    t = terminalreporter
    t.write_sep("=", "NOTE")
    t.write_line("%s\nRandom Seed: %d" % (mdp.config.info(),
                                          t.config.option.seed))
    if 'failed' in t.stats or 'error' in t.stats:
        t.write_line(_err_str)

def pytest_configure(config):
    seed = config.getvalue("seed")
    if seed is None or seed == ('NO', 'DEFAULT'):
        config.option.seed = int(mdp.numx_rand.randint(2**31-1))

def pytest_runtest_setup(item):
    # set random seed
    mdp.numx_rand.seed(item.config.option.seed)

def pytest_addoption(parser):
    """Add random seed option to py.test"""
    parser.addoption("--seed", dest="seed", type="int", action="store",
                     help="set random seed")

def pytest_namespace():
    return dict(mdp_toolkit_configured=True)
