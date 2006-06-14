import glob
import os
from distutils.core import setup

short_description = "Modular toolkit for Data Processing (MDP) is a "+\
                    "Python library to perform data processing. Already"+\
                    " implemented algorithms include: Principal Component"+\
                    " Analysis (PCA), Independent Component Analysis (ICA)"+\
                    ", Slow Feature Analysis (SFA), and Growing Neural Gas"+\
                    " (GNG)." 

long_description = """
Modular toolkit for Data Processing (MDP) is a Python library to
implement data processing elements (nodes) and to combine them into
data processing sequences (flows).

A node is the basic unit in MDP, and it represents a data processing
element, like for example a learning algorithm, a filter, a
visualization step etc. Each node can have a training phase, during
which the internal structures are learned from training data (e.g. the
weights of a neural network are adapted or the covariance matrix is
estimated) and an execution phase, where new data can be processed
forwards (by processing the data through the node) or backwards (by
applying the inverse of the transformation computed by the node if
defined). MDP is designed to make the implementation of new algorithms
easy and intuitive, for example by setting automatically input and
output dimension and by casting the data to match the typecode
(e.g. float or double precision) of the internal structures. Most of
the nodes were designed to be applied to arbitrarily long sets of
data: the internal structures can be updated successively by sending
chunks of the input data (this is equivalent to online learning if the
chunks consists of single observations, or to batch learning if the
whole data is sent in a single chunk). Already implemented nodes
include Principal Component Analysis (PCA), Independent Component
Analysis (ICA), Slow Feature Analysis (SFA), and Growing Neural Gas
Network.

A flow consists in an acyclic graph of nodes (currently only node
sequences are implemented). The data is sent to an input node and is
successively processed by the following nodes on the graph. The
general flow implementation automatizes the training, execution and
inverse execution (if defined) of the whole graph. Crash recovery is
optionally available: in case of failure, the current state of the flow
is saved for later inspection. A subclass of the basic flow class
allows user-supplied checkpoint functions to be executed at the end of
each phase, for example to save the internal structures of a node for
later analysis.

MDP supports the most common numerical extensions to Python and the
symeig package (a Python wrapper for the LAPACK functions to solve
the standard and generalized eigenvalue problems for symmetric
(hermitian) positive definite matrices). MDP also includes graph
(a lightweight package to handle graphs).

When used together with SciPy (the scientific Python library) and symeig,
MDP gives to the scientific programmer the full power of well-known C and
FORTRAN data processing libraries.  MDP helps the programmer to
exploit Python object oriented design with C and FORTRAN efficiency.

MDP has been written for research in neuroscience, but it has been
designed to be helpful in any context where trainable data processing
algorithms are used.  Its simplicity on the user side together with
the reusability of the implemented nodes could make it also a valid
educational tool.

http://mdp-toolkit.sourceforge.net
"""

classifiers = ["Development Status :: 5 - Production/Stable",
               "Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU Library or "+\
               "Lesser General Public License (LGPL)",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics",
               "Topic :: Software Development"]

setup(name = 'MDP', version = '1.1.1',
      author = 'Pietro Berkes and Tiziano Zito',
      author_email = '{p.berkes,t.zito}@biologie.hu-berlin.de',
      maintainer = 'Pietro Berkes and Tiziano Zito',
      maintainer_email = '{p.berkes,t.zito}@biologie.hu-berlin.de',
      license = "http://www.gnu.org/copyleft/lesser.html",
      platforms = ["Any"],
      url = 'http://mdp-toolkit.sourceforge.net',
      download_url = 'http://sourceforge.net/project/showfiles.php?group_id=116959',
      description = short_description,
      long_description = long_description,
      classifiers = classifiers,
      packages = ['mdp', 'mdp.nodes', 'mdp.utils',
                  'mdp.test', 'mdp.demo', 'graph', 'graph.test'],
	  )
