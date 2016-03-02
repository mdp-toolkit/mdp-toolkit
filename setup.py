from setuptools import setup
from setuptools.command.test import test as _test
import os
import sys

email = 'mdp-toolkit-devel@lists.sourceforge.net'

classifiers = ["Development Status :: 5 - Production/Stable",
               "Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]

def get_module_code():
    # keep old python compatibility, so no context managers
    mdp_init = open(os.path.join(os.getcwd(), 'mdp', '__init__.py'))
    module_code = mdp_init.read()
    mdp_init.close()
    return module_code

def throw_bug():
    raise ValueError('Can not get MDP version!\n'
                     'Please report a bug to ' + email)

import ast

def get_extract_variable(tree, variable):
    for node in ast.walk(tree):
        if type(node) is ast.Assign:
            try:
                if node.targets[0].id == variable:
                    return node.value.s
            except:
                pass
    throw_bug()

def get_mdp_ast_tree():
    return ast.parse(get_module_code())

def get_version():
    tree = get_mdp_ast_tree()
    return get_extract_variable(tree, '__version__')

def get_short_description():
    tree = get_mdp_ast_tree()
    return get_extract_variable(tree, '__short_description__')

def get_long_description():
    tree = get_mdp_ast_tree()
    return ast.get_docstring(tree)

class MDPTest(_test):
    def run_tests(self):
        import mdp
        import bimdp
        # Fix random seed here, as we want reproducible failures in
        # automatic builds using "python setup.py test"
        # If the tests are run manually with py.test or
        # using the mdp.test and bimdp.test functions, the seed
        # is not set
        errno = mdp.test(seed=725021957)
        errno += bimdp.test(seed=725021957)
        sys.exit(errno)

def setup_package():

    # check that we have a version
    version = get_version()
    short_description = get_short_description()
    long_description = get_long_description()
    # create download url:
    dl = ('http://sourceforge.net/projects/mdp-toolkit/files/mdp-toolkit/' +
          get_version()+'/MDP-'+get_version()+'.tar.gz')

    setup(name = 'MDP', version=version,
          author = 'MDP Developers',
          author_email = email,
          maintainer = 'MDP Developers',
          maintainer_email = email,
          license = "http://mdp-toolkit.sourceforge.net/license.html",
          platforms = ["Any"],
          url = 'http://mdp-toolkit.sourceforge.net',
          download_url = dl,
          description = short_description,
          long_description = long_description,
          classifiers = classifiers,
          packages = ['mdp', 'mdp.nodes', 'mdp.utils', 'mdp.hinet',
                      'mdp.test', 'mdp.graph', 'mdp.caching',
                      'mdp.parallel', 'bimdp', 'bimdp.hinet', 'bimdp.inspection',
                      'bimdp.nodes', 'bimdp.parallel', 'bimdp.test'],
          package_data = {'mdp.hinet': ['hinet.css'],
                          'mdp.utils': ['slideshow.css']},
          install_requires = ['numpy', 'future'],
          tests_require = ['pytest'],
          # define optional dependencies here, so that they can be installed
          # for example using the "pip -e MDP[scipy] syntax"
          extras_require = {'pp' : 'pp',
                            'joblib' : 'joblib',
                            'scikit-learn' : 'scikit-learn',
                            'scipy' : 'scipy'},
          cmdclass = {'test': MDPTest}
          )


if __name__ == '__main__':
    setup_package()
