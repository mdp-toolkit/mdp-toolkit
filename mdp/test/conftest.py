# global hooks for py.test
import mdp
import py.test

_err_str = """
IMPORTANT: some tests use random numbers. This could
occasionally lead to failures due to numerical degeneracies.
To rule this out, please run the tests more than once.
If you get reproducible failures please report a bug!
"""

def _have_option(parser, optionname):
    return any(optionname == option.get_opt_string()
               for option in parser._anonymous.options)

def pytest_configure(config):
    seed = config.getvalue("seed")
    if seed is None or seed == ('NO', 'DEFAULT'):
        config.option.seed = int(mdp.numx_rand.randint(2**31-1))

def pytest_runtest_setup(item):
    # set random seed
    mdp.numx_rand.seed(item.config.option.seed)

def pytest_addoption(parser):
    """Add random seed option to py.test if it isn't already there
    """
    if not _have_option(parser, '--seed'):
        parser.addoption('--seed', dest='seed', type=int, action='store',
                         help='set random seed')

try:
    py.test.mdp_toolkit_reporting_configured
except AttributeError:
    # Global variable py.test.mdp_toolkit_reporting_configured is necessary
    # to not create the pytest_* functions. Otherwise the terminal
    # report would be printed twice: once for mdp/ and the second time
    # for bimdp/test. When running tests just for one of the
    # directories, the report is still printed correctly.
    def pytest_report_header(config):
        return '%s\nRandom Seed: %d\n' % (mdp.config.info(), config.option.seed)

    def pytest_terminal_summary(terminalreporter):
        t = terminalreporter
        t.write_sep("=", "NOTE")
        t.write_line("%s\nRandom Seed: %d" % (mdp.config.info(),
                                              t.config.option.seed))
        if 'failed' in t.stats or 'error' in t.stats:
            t.write_line(_err_str)

    def pytest_namespace():
        return dict(mdp_toolkit_reporting_configured=True)
