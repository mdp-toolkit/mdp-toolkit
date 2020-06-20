<a href="https://mdp-toolkit.github.io/">
  <img align="right" width="280" height="140" src="https://mdp-toolkit.github.io/_images/logo_animation.gif"><br>
</a>


[![PyPI Latest Release](https://img.shields.io/pypi/v/mdp.svg)](https://pypi.org/project/MDP/)
[![Package Status](https://img.shields.io/pypi/status/mdp.svg)](https://pypi.org/project/MDP/)
[![Downloads](https://img.shields.io/pypi/dm/mdp)](https://mdp-toolkit.github.io/)
[![Downloads](https://img.shields.io/badge/go%20to-MDP's%20website-red)](https://mdp-toolkit.github.io/)

# Modular toolkit for Data Processing

**Modular toolkit for Data Processing (MDP)** is a Python data processing framework.

From the user’s perspective, MDP is a collection of supervised and unsupervised learning algorithms and other data processing units that can be combined into data processing sequences and more complex feed-forward network architectures.

From the scientific developer’s perspective, MDP is a modular framework, which can easily be expanded. The implementation of new algorithms is easy and intuitive. The new implemented units are then automatically integrated with the rest of the library.

- **Website (including documentation):** https://mdp-toolkit.github.io/
- **Mailing list:** https://mail.python.org/mm3/mailman3/lists/mdp-toolkit.python.org/
- **Source:** https://github.com/mdp-toolkit/mdp-toolkit/
- **Bug reports:** https://github.com/mdp-toolkit/mdp-toolkit/issues

#### Main features

The base of available algorithms is steadily increasing and includes

* signal processing methods such as
  * Independent Component Analysis,
  * Principal Component Analysis, and
  * Slow Feature Analysis;
* manifold learning methods such as the [Hessian] Locally Linear Embedding;
* several classifiers;
* probabilistic methods such as
  * Factor Analysis,
  * Fisher Discriminant Analysis,
  * Linear Regression, and
  * RBMs;
* data pre-processing methods such as
  * expansion methods for feature generation and
  * whitening for data normalization;
  
and many others.

You can find out more about MDP's functionality in the [node list](https://mdp-toolkit.github.io/node_list.html) and the [utilities description](https://mdp-toolkit.github.io/additional_utilities.html).

#### Install the newest release

MDP is listed in the [Python Package Index](http://pypi.python.org/pypi/MDP) and can be
installed with:
```sh
pip install mdp
```

#### Install the development version

If you want to live on the bleeding edge, install the development version from the repository with:
```sh
pip install git+https://github.com/mdp-toolkit/mdp-toolkit.git
```

#### Usage

Using MDP is as easy as:
```python
import mdp

# perform PCA on some data x
y = mdp.pca(x)

# perform ICA on some data x using single precision
y = mdp.fastica(x, dtype='float32')
```

## [Documentation](https://mdp-toolkit.github.io/documentation.html)


## Contact and development

MDP has been originally written by [Pietro Berkes](http://people.brandeis.edu/~berkes/) and [Tiziano Zito](https://github.com/otizonaizit) at the [Institute for Theoretical Biology](http://itb.biologie.hu-berlin.de/) of the [Humboldt University](http://www.hu-berlin.de/), Berlin in 2003.

Since 2017, MDP is primarily maintained by the research group [Theory of Neural Systems](https://www.ini.rub.de/research/groups/theory_of_neural_systems/) at the [Institute for Neural Computation](https://www.ini.rub.de/) of the [Ruhr University Bochum](https://www.ruhr-uni-bochum.de/en).

#### Contact

Most development discussions take place in this repository on Github. You are also encouraged to get in touch with the developers and other users on the users’ [mailing list](https://mail.python.org/mm3/mailman3/lists/mdp-toolkit.python.org/).

#### Contributing

MDP is open to user contributions. Users have already contributed some of the nodes, and more contributions are currently being reviewed for inclusion in future releases of the package

If you want to commit code, it may be easiest to fork the MDP repository on github and give us a note on the mailing list. We may then discuss how to integrate your modifications. For simple fixes that don’t need much discussion, you can also send a mail patch to the list using git format-patch or similar.

Your code contribution should not have any additional dependencies, i.e. they should require only the numpy module to be installed. If your code requires some other module, e.g. scipy or C/C++ compilation, ask mdp-toolkit@python.org for assistance.

To learn more about how to contribute to MDP, check out the [information for new developers section](https://mdp-toolkit.github.io/development.html#information-for-new-developers) on the MDP webpage.


## How to cite MDP

If you use MDP for scientific purposes, you may want to cite it. This is the
official way to do it:

Zito, T., Wilbert, N., Wiskott, L., Berkes, P. (2009). 
**Modular toolkit for Data Processing (MDP): a Python data processing frame
work**, Front. Neuroinform. (2008) **2**:8. [doi:10.3389/neuro.11.008.2008](http://www.frontiersin.org/neuroinformatics/10.3389/neuro.11.008.2008/full).

If your paper gets published, please send us a reference (and even a copy if
you don't mind).
