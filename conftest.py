collect_ignore = ['build', 'cover', 'html',
                  ]

def pytest_addoption(parser):
    """Add random seed option to py.test
    """
    parser.addoption('--seed', dest='seed', type=int, action='store',
                     help='set random seed')
