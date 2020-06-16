<div align="center">
  <img align="right" width="200" height="100" src="https://mdp-toolkit.github.io/_images/logo_animation.gif"><br>
</div>

# Modular Toolkit for Data Processing

**Modular toolkit for Data Processing (MDP)** is a Python data processing framework.

From the user’s perspective, MDP is a collection of supervised and unsupervised learning algorithms and other data processing units that can be combined into data processing sequences and more complex feed-forward network architectures.

From the scientific developer’s perspective, MDP is a modular framework, which can easily be expanded. The implementation of new algorithms is easy and intuitive. The new implemented units are then automatically integrated with the rest of the library.

## Main Features

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

You can find out more about MDP's functionality in the [node list](https://mdp-toolkit.github.io/node_list.html) and [the utilities description](https://mdp-toolkit.github.io/additional_utilities.html).

## Installation

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

Please refer to the online documentation at https://mdpdocs.readthedocs.io
The legacy documentation is still available at http://mdp-toolkit.sourceforge.net
The source code is available at https://github.com/mdp-toolkit/mdp-toolkit
