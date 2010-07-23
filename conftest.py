collect_ignore = ['build', 'cover', 'html',
                  'mdp/parallel/pp_test',
                  ]

def pytest_addoption(parser):
    """Add random seed option to py.test
    """
    parser.addoption('--seed', dest='seed', type=int, action='store',
                     help='set random seed')
