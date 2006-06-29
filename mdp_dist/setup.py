import glob
import os
from distutils.core import setup

short_description = "MDP is a Python data processing framework. Implemented algorithms include: Principal Component Analysis, Independent Component Analysis, Slow Feature Analysis, Growing Neural Gas, Factor Analysis, Fisher Discriminant Analysis, and Gaussian Classifiers."

long_description = """
Modular toolkit for Data Processing (MDP) is a data processing
framework written in Python.

From the user's perspective, MDP consists of a collection of trainable
supervised and unsupervised algorithms or other data processing units
(nodes) that can be combined into data processing flows. Given a
sequence of input data, MDP takes care of successively training or
executing all nodes in the flow. This structure allows to specify
complex algorithms as a sequence of simpler data processing steps in a
natural way. Training can be performed using small chunks of input
data, so that the use of very large data sets becomes possible while
reducing the memory requirements. Memory usage can also be minimized
by defining the internals of the nodes to be single precision.

The base of readily available algorithms includes Principal Component
Analysis, two flavors of Independent Component Analysis, Slow Feature
Analysis, Gaussian Classifiers, Growing Neural Gas, Fisher
Discriminant Analysis, and Factor Analysis.

From the developer's perspective, MDP is a framework to make the
implementation of new algorithms easier. The basic class 'Node' takes
care of tedious tasks like numerical type and dimensionality checking,
leaving the developer free to concentrate on the implementation of the
training and execution phases. The node then automatically integrates
with the rest of the library and can be used in a flow together with
other nodes. A node can have multiple training phases and even an
undetermined number of phases. This allows for example the
implementation of algorithms that need to collect some statistics on
the whole input before proceeding with the actual training, or others
that need to iterate over a training phase until a convergence
criterion is satisfied. The ability to train each phase using chunks
of input data is maintained if the chunks are generated with
iterators. Moreover, crash recovery is optionally available: in case
of failure, the current state of the flow is saved for later
inspection.

MDP has been written in the context of theoretical research in
neuroscience, but it has been designed to be helpful in any context
where trainable data processing algorithms are used. Its simplicity on
the user side together with the reusability of the implemented nodes
make it also a valid educational tool.

As its user base is steadily increasing, MDP appears as a good
candidate for becoming a common repository of user-supplied, freely
available, Python implemented data processing algorithms.

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
               "Topic :: Software Development :: Algorithms"]

setup(name = 'MDP', version = '2.0RC',
      author = 'Pietro Berkes and Tiziano Zito',
      author_email = 'berkes@gatsby.ucl.ac.uk, t.zito@biologie.hu-berlin.de',
      maintainer = 'Pietro Berkes and Tiziano Zito',
      maintainer_email = 'berkes@gatsby.ucl.ac.uk, t.zito@biologie.hu-berlin.de',
      license = "http://www.gnu.org/copyleft/lesser.html",
      platforms = ["Any"],
      url = 'http://mdp-toolkit.sourceforge.net',
      download_url = 'http://sourceforge.net/project/showfiles.php?group_id=116959',
      description = short_description,
      long_description = long_description,
      classifiers = classifiers,
      packages = ['mdp', 'mdp.nodes', 'mdp.utils',
                  'mdp.test', 'mdp.demo', 'mdp.graph'],
      )
