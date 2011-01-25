from distutils.core import setup
import os
import sys
import ast

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

def get_extract_variable(tree, variable):
    for node in ast.walk(tree):
        if type(node) is ast.Assign:
            try:
                if node.targets[0].id == variable:
                    return node.value.s
            except:
                pass
    raise ValueError('Can not get MDP version!\n'
                     'Please report a bug to %s' % email)

def get_mdp_ast_tree():
    mdp_init = open(os.path.join(os.getcwd(), 'mdp', '__init__.py'))
    module_code = mdp_init.read()
    mdp_init.close()
    return ast.parse(module_code)

def get_version():
    tree = get_mdp_ast_tree()
    return get_extract_variable(tree, '__version__')

def get_short_description():
    tree = get_mdp_ast_tree()
    return get_extract_variable(tree, '__short_description__')

def get_long_description():
    tree = get_mdp_ast_tree()
    return ast.get_docstring(tree)

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
          download_url = 'http://sourceforge.net/projects/mdp-toolkit/files',
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
