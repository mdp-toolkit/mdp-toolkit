# This file must be runnable with all supported python versions:
#    2.5, 2.6, 2.7, 3.1, and 3.2.
# Things which might not be available:
#    context managers, the print statement, some modules (e.g. ast).

from distutils.core import setup
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

try:
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
except ImportError:
    import re

    def get_variable(pattern):
        m = re.search(pattern, get_module_code(), re.M + re.S + re.X)
        if not m:
            throw_bug()
        return m.group(1)

    def get_version():
        return get_variable(r'^__version__\s*=\s*[\'"](.+?)[\'"]')

    def get_short_description():
        text = get_variable(r'''^__short_description__\s*=\s*  # variable name and =
                            \\?\s*(?:"""|\'\'\')\\?\s*         # opening quote with backslash
                            (.+?)
                            \s*(?:"""|\'\'\')''')              # closing quote
        return text.replace(' \\\n', ' ')

    def get_long_description():
        return get_variable(r'''^(?:"""|\'\'\')\\?\s*          # opening quote with backslash
                            (.+?)
                            \s*(?:"""|\'\'\')''')              # closing quote


def setup_package():

    # Perform 2to3 if needed
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    if sys.version_info[0] == 3:
        src_path = os.path.join(local_path, 'build', 'py3k')
        import py3tool
        print("Converting to Python3 via 2to3...")
        py3tool.sync_2to3('mdp', os.path.join(src_path, 'mdp'))
        py3tool.sync_2to3('bimdp', os.path.join(src_path, 'bimdp'))

    # check that we have a version
    version = get_version()
    short_description = get_short_description()
    long_description = get_long_description()
    # create download url:
    dl = ('http://sourceforge.net/projects/mdp-toolkit/files/mdp-toolkit/' +
          get_version()+'/MDP-'+get_version()+'.tar.gz')
    # Run build
    os.chdir(src_path)
    sys.path.insert(0, src_path)

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
                          'mdp.utils': ['slideshow.css']}
          )


if __name__ == '__main__':
    setup_package()
