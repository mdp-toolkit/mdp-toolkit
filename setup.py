from distutils.core import setup
import os
import sys

email = 'mdp-toolkit-devel@lists.sourceforge.net'

short_description = ("Modular toolkit for Data Processing (MDP) is a"
                     " library of widely used data processing algorithms"
                     " that can be combined according to a pipeline"
                     " analogy to build more complex data processing"
                     " software. Implemented algorithms include"
                     " Principal Component Analysis (PCA), Independent"
                     " Component Analysis (ICA), Slow Feature Analysis"
                     " (SFA), and many more.")

long_description = """
The Modular toolkit for Data Processing (MDP) is a library of widely
used data processing algorithms that can be combined according to a
pipeline analogy to build more complex data processing software.

From the user's perspective, MDP consists of a collection of
supervised and unsupervised learning algorithms, and other data
processing units (nodes) that can be combined into data processing
sequences (flows) and more complex feed-forward network
architectures. Given a set of input data, MDP takes care of
successively training or executing all nodes in the network. This
allows the user to specify complex algorithms as a series of simpler
data processing steps in a natural way.

The base of available algorithms is steadily increasing and includes,
to name but the most common, Principal Component Analysis (PCA and
NIPALS), several Independent Component Analysis algorithms (CuBICA,
FastICA, TDSEP, JADE, and XSFA), Slow Feature Analysis, Gaussian
Classifiers, Restricted Boltzmann Machine, and Locally Linear
Embedding.

Particular care has been taken to make computations efficient in terms
of speed and memory.  To reduce memory requirements, it is possible to
perform learning using batches of data, and to define the internal
parameters of the nodes to be single precision, which makes the usage
of very large data sets possible.  Moreover, the 'parallel' subpackage
offers a parallel implementation of the basic nodes and flows.

From the developer's perspective, MDP is a framework that makes the
implementation of new supervised and unsupervised learning algorithms
easy and straightforward.  The basic class, 'Node', takes care of
tedious tasks like numerical type and dimensionality checking, leaving
the developer free to concentrate on the implementation of the
learning and execution phases. Because of the common interface, the
node then automatically integrates with the rest of the library and
can be used in a network together with other nodes. A node can have
multiple training phases and even an undetermined number of phases.
This allows the implementation of algorithms that need to collect some
statistics on the whole input before proceeding with the actual
training, and others that need to iterate over a training phase until
a convergence criterion is satisfied. The ability to train each phase
using chunks of input data is maintained if the chunks are generated
with iterators. Moreover, crash recovery is optionally available: in
case of failure, the current state of the flow is saved for later
inspection.

MDP has been written in the context of theoretical research in
neuroscience, but it has been designed to be helpful in any context
where trainable data processing algorithms are used. Its simplicity on
the user side together with the reusability of the implemented nodes
make it also a valid educational tool.

http://mdp-toolkit.sourceforge.net
"""

classifiers = ["Development Status :: 5 - Production/Stable",
               "Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License"
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]


def get_version():
    mdp_init = file(os.path.join(os.getcwd(), 'mdp', '__init__.py'), 'r')
    for line in mdp_init:
        str_ = line.strip()
        if str_.startswith('__version__'):
            exec(str_)
            return __version__
    return None

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
    try:
        version = get_version()
        if not isinstance(version, str) or len(version) == 0:
            raise Exception
    except Exception, exc:
        raise Exception(str(exc) +
                        ('\nCan not get MDP version!\nPlease report '
                         'a bug to %s'%email))
    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    setup(name = 'MDP', version = get_version(),
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
