from builtins import range
import mdp
from .mca_nodes_online import MCANode
from .pca_nodes_online import CCIPCAWhiteningNode as WhiteningNode
from .stats_nodes_online import OnlineCenteringNode, OnlineTimeDiffNode
from mdp.utils import mult, pinv


class IncSFANode(mdp.OnlineNode):
    """
    Incremental Slow Feature Analysis (IncSFA) extracts the slowly varying
    components from the input data incrementally.


    .. attribute:: sf
    
         Slow feature vectors

    .. attribute:: wv
    
         Whitening vectors

    .. attribute:: sf_change
    
         Difference in slow features after update
         
    .. admonition:: Reference
    
        More information about IncSFA
        can be found in Kompella V.R, Luciw M. and Schmidhuber J., Incremental Slow
        Feature Analysis: Adaptive Low-Complexity Slow Feature Updating from
        High-Dimensional Input Streams, Neural Computation, 2012.
    """

    def __init__(self, eps=0.05, whitening_output_dim=None, remove_mean=True, avg_n=None, amn_params=(20, 200, 2000, 3),
                 init_pca_vectors=None, init_mca_vectors=None, input_dim=None, output_dim=None, dtype=None,
                 numx_rng=None):
        """Initialize an object of type 'SFANode'.
        
        :param eps: Learning rate (default: 0.1)
        :type eps: float
        
        :param whitening_output_dim: Whitening output dimension. (default: input_dim)
        :type whitening_output_dim: int
        
        :param remove_mean: Remove input mean incrementally (default: True)
        :type remove_mean: bool
        
        :param avg_n: When set, the node updates an exponential weighted moving average.
            avg_n intuitively denotes a window size. For a large avg_n, avg_n samples
            represents about 86% of the total weight. (Default:None)
        
        :param amn_params: PCA amnesic parameters. Default set to (n1=20,n2=200,m=2000,c=3).
            For n < n1, ~ moving average.
            For n1 < n < n2 - Transitions from moving average to amnesia. m denotes the scaling
            param and c typically should be between (2-4). Higher values will weigh recent data.
        :type amn_params: tuple
        
        :param init_pca_vectors: Initial whitening vectors. Default - randomly set
        :type init_pca_vectors: numpy.ndarray
        
        :param init_mca_vectors: Initial mca vectors. Default - randomly set
        :type init_mca_vectors: numpy.ndarray
        
        :param input_dim: The input dimensionality.
        :type input_dim: int
        
        :param output_dim: The output dimensionality.
        :type output_dim: int
        
        :param dtype: The datatype.
        :type dtype: numpy.dtype or str
        
        :param numx_rng: Random number generator. (Optional)
        """

        self.whiteningnode = WhiteningNode(amn_params=amn_params, init_eigen_vectors=init_pca_vectors,
                                           input_dim=input_dim, output_dim=whitening_output_dim, dtype=dtype,
                                           numx_rng=numx_rng,)
        self.tdiffnode = OnlineTimeDiffNode(dtype=dtype, numx_rng=numx_rng)

        self.mcanode = MCANode(eps=eps, init_eigen_vectors=init_mca_vectors, input_dim=whitening_output_dim,
                               output_dim=output_dim, dtype=dtype, numx_rng=numx_rng)
        if remove_mean:
            self.avgnode = OnlineCenteringNode(avg_n=avg_n, dtype=dtype, numx_rng=numx_rng)

        self.eps = eps
        self.whitening_output_dim = whitening_output_dim
        self.remove_mean = remove_mean
        self.avg_n = avg_n

        super(IncSFANode, self).__init__(input_dim, output_dim, dtype, numx_rng)

        self._new_episode = None
        self._init_sf = None
        self.wv = None
        self.sf = None
        self.sf_change = 0.

    def _set_input_dim(self, n):
        self._input_dim = n
        self.whiteningnode.input_dim = n

    def _set_output_dim(self, n):
        self._output_dim = n
        self.mcanode.output_dim = n

    def _set_dtype(self, t):
        self._dtype = t
        self.whiteningnode.dtype = t
        self.tdiffnode.dtype = t
        self.mcanode.dtype = t
        if self.remove_mean:
            self.avgnode.dtype = t

    def _set_numx_rng(self, rng):
        # set a shared numx rng
        self._numx_rng = rng
        self.whiteningnode.numx_rng = rng
        self.tdiffnode.numx_rng = rng
        self.mcanode.numx_rng = rng
        if self.remove_mean:
            self.avgnode.numx_rng = rng

    @property
    def init_slow_features(self):
        """Return the initialized slow features.
        
        :return: Initialized slow features."""
        return self._init_sf

    @property
    def init_pca_vectors(self):
        """Return the initialized whitening vectors.
        
        :return: Initialized whitening vectors.
        :rtype: numpy.ndarray
        """
        return self.whiteningnode.init_eigen_vectors

    @property
    def init_mca_vectors(self):
        """Return initialized minor components.
        
        :return: Initialized minor components.
        :rtype: numpy.ndarray
        """
        return self.mcanode.init_eigen_vectors

    def _check_params(self, x):
        """Initialize parameters."""
        if self._init_sf is None:
            if self.remove_mean:
                self._pseudo_check_fn(self.avgnode, x)
            x = self.avgnode.execute(x)
            self._pseudo_check_fn(self.whiteningnode, x)
            x = self.whiteningnode.execute(x)
            self._pseudo_check_fn(self.tdiffnode, x)
            self._pseudo_check_fn(self.mcanode, x)
            self._init_sf = mult(self.whiteningnode.init_eigen_vectors, self.mcanode.init_eigen_vectors)
            self.sf = self._init_sf
            if self.output_dim is None:
                self.output_dim = self.mcanode.output_dim

        if self._new_episode is None:  # Set new_episode to True for the very first sample
            self._new_episode = True
        elif self._new_episode:  # Set new_episode to False for the subsequent samples
            self._new_episode = False

    @staticmethod
    def _pseudo_check_fn(node, x):
        node._check_input(x)
        node._check_params(x)

    @staticmethod
    def _pseudo_train_fn(node, x):
        node._train(x)
        node._train_iteration += x.shape[0]

    def _check_train_args(self, x, *args, **kwargs):
        if self.training_type is 'batch':
            # check that we have at least 2 time samples for batch training
            if x.shape[0] < 2:
                raise mdp.TrainingException(
                    "Need at least 2 time samples for 'batch' training type (%d given)" % (x.shape[0]))

    def _step_train(self, x):
        if self.remove_mean:
            self._pseudo_train_fn(self.avgnode, x)
            x = self.avgnode._execute(x)

        self._pseudo_train_fn(self.whiteningnode, x)
        x = self.whiteningnode._execute(x)

        self._pseudo_train_fn(self.tdiffnode, x)

        if self._new_episode:
            return

        x = self.tdiffnode._execute(x)

        self._pseudo_train_fn(self.mcanode, x)

        sf = mult(self.whiteningnode.v, self.mcanode.v)
        sf_change = mdp.numx_linalg.norm(sf - self.sf)
        self.sf = sf
        return sf_change

    def _train(self, x, new_episode=None):
        """Update slow features.
        
        :param new_episode: Set new_episode to True to ignore taking erroneous
            derivatives between the episodes of training data.
        :type new_episode: bool
        """
        sf_change = 0.0
        if self.training_type == 'batch':
            self._new_episode = True
            for i in range(x.shape[0]):
                sf_change = self._step_train(x[i:i + 1])
                self._new_episode = False
        else:
            if new_episode is not None:
                self._new_episode = new_episode
            sf_change = self._step_train(x)

        self.wv = self.whiteningnode.v
        self.sf_change = sf_change

    def _execute(self, x):
        """Return slow feature response.
        
        :return: Slow feature response.
        """
        if self.remove_mean:
            x = self.avgnode._execute(x)
        return mult(x, self.sf)

    def _inverse(self, y):
        """Return inverse of the slow feature response.
        
        :return: The inverse of the slow feature response.
        """
        return mult(y, pinv(self.sf)) + self.avgnode.avg

    def __repr__(self):
        """Print all args.

        :return: A string that contains all argument names and their values.
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
        whit_dim = "whitening_output_dim=%s" % str(self.whitening_output_dim)
        remove_mean = "remove_mean=%s" % str(self.remove_mean)
        avg_n = "avg_n=%s" % self.avg_n
        amn = "\namn_params=%s" % str(self.whiteningnode.amn_params)
        init_pca_vecs = "init_pca_vectors=%s" % str(self.init_pca_vectors)
        init_mca_vecs = "init_pca_vectors=%s" % str(self.init_mca_vectors)
        args = ', '.join(
            (eps, whit_dim, remove_mean, avg_n, amn, init_pca_vecs, init_mca_vecs, inp, out, typ, numx_rng))
        return name + '(' + args + ')'

