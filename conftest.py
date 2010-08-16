collect_ignore = ['build', 'cover', 'html',
                  ]

def _have_option(parser, optionname):
    return any(optionname == option.get_opt_string()
               for option in parser._anonymous.options)

def pytest_addoption(parser):
    """Add random seed option to py.test
    """
    if not _have_option(parser, '--seed'):
        parser.addoption('--seed', dest='seed', type=int, action='store',
                         help='set random seed')
