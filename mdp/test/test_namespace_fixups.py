import sys
from _tools import *

def list_dunder_module_dunder(module):
    try:
        names = module.__all__
    except AttributeError:
        names = dir(module)
    for name in names:
        if name.startswith('_'):
            continue
        item = getattr(module, name)
        try:
            modname = getattr(item, '__module__')
        except AttributeError:
            continue
        if hasattr(item, '__module__'):
            yield modname, name, item

MODULES = ['mdp',
           'mdp.nodes',
           'mdp.hinet',
           'mdp.parallel',
           'mdp.graph',
           'mdp.utils',
           ]

def pytest_generate_tests(metafunc):
    generate_calls(MODULES, metafunc)

def generate_calls(modules, metafunc):
    for module in modules:
        metafunc.addcall(funcargs=dict(parentname=module), id=module)

def test_dunder_module_dunder(parentname):
    rootname = parentname.split('.')[-1]
    module = sys.modules[parentname]
    for modname, itemname, item in list_dunder_module_dunder(module):
        parts = modname.split('.')
        assert (parts[0] != rootname or
                modname == parentname), \
                '{0}.{1}.__module_ == {2.__module__} != {3}' \
                .format(parentname, itemname, item, parentname)
