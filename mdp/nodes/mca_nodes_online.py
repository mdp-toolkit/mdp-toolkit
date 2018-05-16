from builtins import range
import mdp
from mdp.utils import mult
from past.utils import old_div


class MCANode(mdp.OnlineNode):
    """
    Minor Component Analysis (MCA) extracts minor components (dual of principal
    components) from the input data incrementally.

    .. attribute:: v
    
         Eigen vectors

    .. attribute:: d
    
         Eigen values
    
    |
    
    .. admonition:: Reference
        
        More information about MCA can be found in
        Peng, D. and Yi, Z, A new algorithm for sequential minor component
        analysis, International Journal of Computational Intelligence Research,
        2(2):207--215, 2006.


    """

    def __init__(self, eps=0.1, gamma=1.0, normalize=True, init_eigen_vectors=None, input_dim=None, output_dim=None,
                 dtype=None, numx_rng=None):
        """Initializes an object of type 'MCANode'.
        
        :param eps: Learning rate (default: 0.1).
        :type eps: float
        
        :param gamma: Sequential addition coefficient (default: 1.0).
        :type gamma: float
        
        :param normalize: If True, eigenvectors are normalized after every update.
            Useful for non-stationary input data.  (default: True)
        :type normalize: bool
        
        :param init_eigen_vectors: initial eigen vectors. Default - randomly set
        :type init_eigen_vectors: numpy.ndarray
        
        :param input_dim: The input dimensionality.
        :type input_dim: int
        
        :param output_dim: The output dimensionality.
        :type output_dim: int
        
        :param dtype: The datatype.
        :type dtype: numpy.dtype or str
        
        :param numx_rng: OnlineNodes support use of a pre-seeded random number generator
            through a 'numx_rng' argument. This can be useful to replicate
            results.
        """
        super(MCANode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.eps = eps
        self.gamma = gamma
        self.normalize = normalize

        self._init_v = init_eigen_vectors

        self.v = None  # Eigenvectors
        self.d = None  # Eigenvalues

    @property
    def init_eigen_vectors(self):
        """Return initialized eigenvectors (minor components).
        
        :return: The initialized eigenvectors.
        :rtype: numpy.ndarray
        """
        return self._init_v

    @init_eigen_vectors.setter
    def init_eigen_vectors(self, init_eigen_vectors=None):
        """Set initial eigen vectors (minor components).
        
        :param init_eigen_vectors: The inital eigenvectors which are set before the
            online routine starts.
        :type init_eigen_vectors: numpy.ndarray
        """
        self._init_v = init_eigen_vectors
        if self._input_dim is None:
            self._input_dim = self._init_v.shape[0]
        else:
            assert (self.input_dim == self._init_v.shape[0]), mdp.NodeException(
                'Dimension mismatch. init_eigen_vectors '
                'shape[0] must be'
                '%d, given %d' % (self.input_dim,
                                  self._init_v.shape[0]))
        if self._output_dim is None:
            self._output_dim = self._init_v.shape[1]
        else:
            assert (self.output_dim == self._init_v.shape[1]), mdp.NodeException(
                'Dimension mismatch. init_eigen_vectors'
                ' shape[1] must be'
                '%d, given %d' % (self.output_dim,
                                  self._init_v.shape[1]))
        if self.v is None:
            self.v = self._init_v.copy()
            self.d = mdp.numx.sum(self.v ** 2, axis=0) ** 0.5  # identical with np.linalg.norm(self.v, axis=0)
            # Using this for backward numpy (versions below 1.8) compatibility.

    def _check_params(self, *args):
        """Initialize parameters."""
        if self._init_v is None:
            if self.output_dim is not None:
                self.init_eigen_vectors = 0.1 * self.numx_rng.randn(self.input_dim, self.output_dim).astype(self.dtype)
            else:
                self.init_eigen_vectors = 0.1 * self.numx_rng.randn(self.input_dim, self.input_dim).astype(self.dtype)

    def _train(self, x):
        """Update the minor components."""
        c = mult(x.T, x)
        for j in range(self.output_dim):
            v = self.v[:, j:j + 1]
            d = self.d[j]

            n = self.eps / (1 + j * 1.2)
            a = mult(c, v)
            if self.normalize:
                v = (1.5 - n) * v - n * a
            else:
                v = (1.5 - n * (d ** 2)) * v - n * a
            l = mult(v.T, v)
            c += self.gamma * mult(v, v.T) / l

            self.v[:, j:j + 1] = v
            self.d[j] = mdp.numx.sqrt(l)
            if self.normalize:
                self.v[:, j:j + 1] = old_div(v, self.d[j])

    def get_projmatrix(self, transposed=1):
        """Return the projection matrix.
        
        :param transposed: If the projection matrix should return transposed,
            set this to True.
        :type transposed: bool
        
        :return: The projection matrix.
        :rtype: numpy.ndarray
        """
        if transposed:
            return self.v
        return self.v.T

    def get_recmatrix(self, transposed=1):
        """Return the back-projection matrix (i.e. the reconstruction matrix).
        
        :param transposed:  If the back-projection matrix should return
            transposed, set this to True.
        :type transposed: bool
        
        :return: The back-projection matrix.
        :rtype: numpy.ndarray
        """
        if transposed:
            return self.v.T
        return self.v

    def _execute(self, x, n=None):
        """Project the input on the first 'n' principal components.
        
        :param x: The input that is to project.
        :type x: numpy.ndarray
        
        :param n: The number of first principle components to project on.
            If 'n' is not set, use all available components.
        :type n: int
        
        :return: The projected input.
        :rtype: numpy.ndarray
        """
        if n is not None:
            return mult(x, self.v[:, :n])
        return mult(x, self.v)

    def _inverse(self, y, n=None):
        """Project 'y' to the input space using the first 'n' components.
        
        :param y: Vectors from the output space.
        :type y: numpy.ndarray
        
        :param n: The number of components to use for projection to the
            input space. If 'n' is not set, use all available components.
        :type n: int
        
        :return: The projected vectors.
        :rtype: numpy.ndarray
        
        :raises mdp.NodeException: If the valid dimension is exceeded.
        """
        if n is None:
            n = y.shape[1]
        if n > self.output_dim:
            error_str = ("y has dimension %d,"
                         " should be at most %d" % (n, self.output_dim))
            raise mdp.NodeException(error_str)

        v = self.get_recmatrix()
        if n is not None:
            return mult(y, v[:n, :])
        return mult(y, v)

    def __repr__(self):
        """Print all args.
        
        :return: A string that contains all names and their values.
        :rtype: str
        """
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" % self.dtype.name
        numx_rng = "numx_rng=%s" % str(self.numx_rng)
        eps = "\neps=%s" % str(self.eps)
        gamma = "gamma=%s" % str(self.gamma)
        normalize = "normalize=%s" % str(self.normalize)
        init_eig_vecs = "init_eigen_vectors=%s" % str(self.init_eigen_vectors)
        args = ', '.join((eps, gamma, normalize, init_eig_vecs, inp, out, typ, numx_rng))
        return name + '(' + args + ')'
