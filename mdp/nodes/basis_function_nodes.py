import mdp
from mdp.utils import mult
import itertools
from misc_nodes import GridProcessingNode


class BasisFunctionNode(mdp.Node):
    """
    BasisFunctionNode is a non-trainable node that
    provides a set of non-linear basis functions,
    such that, any continuous function can be
    closely approximated as a linear combination
    of these basis functions.

    Supported basis functions are:
    'indicator', 'fourier', 'polynomial', 'radial',
    'lem' (laplacian eigen maps), 'gsfa' (graph-based slow features).

    The node also provides a utility method called 'get_fn_responses_img' that
    returns an image of the basis function responses
    for uniformly selected inputs within the given lims. This is
    useful for visualizing the functions and debugging.

    """

    def __init__(self, basis_name, lims, order=3, decoupled=False, basis_params=None, scale_out=1.0, n_grid_pts=None,
                 input_dim=None, dtype=None):
        """
        basis_name - name of the basis function to use: Supported: 'indicator', 'fourier', 'polynomial', 'radial',
                                'lem' (laplacian eigen maps), 'gsfa' (graph-based slow features).
        lims - a tuple of lower and upper bounds for each dimension of the input.
                Eg., [(lower1, lower2, ...,), (upper1, upper2, ...,)]

        order - order of the basis functions. A value that determines the dimensionality of the space
                that the basis spans.
                        Eg., order 4 of polynomial basis translates x to [1, x, x^2, x^3]
                        Eg., order 30 of LEMs return top 30 laplacian eigen-maps excluding the constant function.
                        Note that the first basis function (a constant function) for LEM and GSFA is filtered out.

        decoupled - When True, basis functions do not include combinations of the input dims.
                        Eg., order 3 of polynomial basis translates x to [1, x, x^2, y, y^2]
                        If False, x -> [1, x, x^2, y, y^2, xy, xy^2, x^2y, x^2y^2].

        basis_params - Additional (required or optional) params (list or a dict) that define the basis.

        scale_out - scale output by a contant. y = basisfn(x)*scale_out

        n_grid_pts - number of grid points within each dimension. A parameter used for discrete inputs.
                     Eg. [n1, n2, ...,]

        """
        super(BasisFunctionNode, self).__init__(input_dim, output_dim=None, dtype=dtype)
        self.basis_name = basis_name

        if len(lims) != 2:
            raise mdp.NodeException("'lims' has %d elements given, required 2 "
                                    "[(lower1, lower2, ...,), (upper1, upper2, ...,)]" % (len(lims)))
        if mdp.numx.isscalar(lims[0]):
            lims = [tuple((lim,)) for lim in lims]
        if len(lims[0]) != len(lims[1]):
            raise mdp.NodeException("Length of lower_bounds ('lims[0]=%d') does not match the length "
                                    "of the upper_bounds ('lims[1]=%d)." % (len(lims[0]), len(lims[1])))
        self.lims = mdp.numx.asarray(lims)

        self.order = order
        self.scale_out = scale_out
        self.decoupled = decoupled
        self.basis_params = basis_params

        self._gp = GridProcessingNode(grid_lims=lims, n_grid_pts=n_grid_pts)

        self._input_dim = len(lims[0])

        self._basis_mode = 'continuous'

        # Basis Fns Init

        if self.basis_name is 'indicator':
            self._basis_mode = 'discrete'
            self._gp.output_type = 'graphindx'
            self._output_dim = mdp.numx.prod(self._gp.n_grid_pts)

        elif self.basis_name is 'fourier':
            iterprod = itertools.product(map(str, mdp.numx.arange(self.order + 1)), repeat=self._input_dim)
            self._coeffs = mdp.numx.array([list(map(int, x)) for x in iterprod])
            self._output_dim = int(mdp.numx.power(self.order + 1.0, self._input_dim))
            if self.decoupled:
                self._coeffs = list(itertools.ifilter(
                    lambda a: mdp.numx.sum(a == 0) == (self._input_dim - 1), self._coeffs))[::-1]
                self._coeffs.append(mdp.numx.zeros(self._input_dim, dtype='int'))
                self._coeffs = self._coeffs[::-1]
                self._output_dim = int(self.order * self._input_dim) + 1
            assert (self._output_dim == len(self._coeffs))

        elif self.basis_name is 'polynomial':
            self._output_dim = int(mdp.numx.power(self.order + 1.0, self._input_dim))
            self._powers = list(itertools.product(mdp.numx.arange(self.order + 1), repeat=self._input_dim))
            if self.decoupled:
                self._output_dim = int(self.order * self._input_dim) + 1
                self._powers = filter(lambda a: (mdp.numx.sum(a) == a).any(), self._powers)
            assert (self._output_dim == len(self._powers))

        elif self.basis_name is 'radial':
            self._centers = list(itertools.product(
                *map(lambda a: mdp.numx.linspace(a[0], a[1], self.order, endpoint=True), zip(*self.lims))))
            self._sigmas = mdp.numx.diag((self.lims[1, :] - self.lims[0, :] + 1) / float(self.order))
            if isinstance(basis_params, list):
                if len(basis_params) > 0:
                    self._centers = basis_params[0]
                if len(basis_params) > 1:
                    self._sigmas = basis_params[1]
            elif isinstance(basis_params, dict):
                self._centers = basis_params.get('centers', self._centers)
                self._sigmas = basis_params.get('sigmas', self._sigmas)
            self._output_dim = int(mdp.numx.power(self.order, self._input_dim))
            self._rbfnode = mdp.nodes.RBFExpansionNode(self._centers, self._sigmas ** 2)

        elif self.basis_name is 'lem':
            opt_args = ['adjacency']
            default_args_str = ['grid_adjacency_matrix']
            self._basis_mode = 'discrete'
            if basis_params is None:
                print ("\n'%s' basis function also takes optional arguments via \n'basis_params' = "
                       "%s (as a list or dict). \nSince the given 'basis_params' in set to None, \ndefault "
                       "values=%s are used." % (basis_name, str(opt_args), str(default_args_str)))
                basis_params = [self._gp.get_adjacency_matrix()]
            if isinstance(basis_params, list):
                self._adj = basis_params[0]
            elif isinstance(basis_params, dict):
                try:
                    self._adj = basis_params["adjacency"]
                except KeyError:
                    raise mdp.NodeException(
                        " Invalid Key! %s Valid 'basis_params' keys are %s." % (basis_name, str(opt_args)))
            try:
                from scipy.linalg import eigh
            except ImportError:
                eigh = mdp.numx_linalg.eigh

            self._gp.output_type = 'graphindx'
            self._output_dim = int(self.order)
            self._L = self._get_laplacian(self._adj)
            [w, v] = eigh(self._L)
            self.v = v[mdp.numx.argsort(w)][:, 1:self._output_dim + 1]

        elif self.basis_name is 'gsfa':
            opt_args = ['adjacency', 'degree', 'n_layers', 'n_poly']
            default_args_str = ['grid_adjacency_matrix', 3, 3, 4]
            if basis_params is None:
                if basis_params is None:
                    print ("\n'%s' basis function also takes optional arguments via \n'basis_params' = "
                           "%s (as a list or dict). \nSince the given 'basis_params' in set to None, \ndefault "
                           "values=%s are used." % (basis_name, str(opt_args), str(default_args_str)))
                basis_params = [self._gp.get_adjacency_matrix(), 3, 3, 4]

            if isinstance(basis_params, list):
                self._adj, self._degree, self._nlayers, self._npoly = basis_params[:4]
            elif isinstance(basis_params, dict):
                try:
                    self._adj = basis_params.get("adjacency", self._gp.get_adjacency_matrix())
                    self._degree = basis_params.get("degree", 3)
                    self._nlayers = basis_params.get("n_layers", 3)
                    self._npoly = basis_params.get("n_poly", 4)
                except KeyError:
                    raise mdp.NodeException(
                        " Invalid Key! %s Valid 'basis_params' keys are %s." % (basis_name, str(opt_args)))

            from mdp.nodes import PolynomialExpansionNode
            try:
                from scipy.linalg import eigh
            except ImportError:
                eigh = mdp.numx_linalg.eigh

            self._gp.output_type = 'graphindx'

            mat_degree = mdp.numx.diag(self._adj.sum(axis=0))
            mat_lapl = self._get_laplacian(self._adj)

            tot_samples = mdp.numx.product(self._gp.n_grid_pts)
            x = self._gp.inverse(mdp.numx.arange(tot_samples)[:, None])
            indices = [self._get_neighbors(self._adj, i) for i in xrange(tot_samples)]
            self.fa = []
            self.v = []
            for layernum in xrange(self._nlayers):
                self.fa.append(PolynomialExpansionNode(self._degree))
                z = self.fa[layernum](x)
                mat_lapl_ = 0
                mat_degree_ = 0
                for i in xrange(z.shape[0]):
                    indx_ = indices[i]
                    za = z[i:i + 1]
                    for i_ in indx_:
                        zb = self.fa[layernum](x[i_:i_ + 1])
                        mat_lapl_ += mult(za.T, mat_lapl[i, i_] * zb)
                        mat_degree_ += mult(za.T, mat_degree[i, i_] * zb)
                [w, v] = eigh(mat_lapl_, mat_degree_)
                self.v.append(v[mdp.numx.argsort(w)])
                x = mult(z, self.v[layernum][:, :min(self.v[layernum].shape[1], self._npoly)])
                x += 0.0001 * mdp.numx_rand.randn(*x.shape)
            self._output_dim = int(min(self.order, self.v[-1].shape[0]))
            self._gp.output_type = 'gridx'
        else:
            # subclasses can add additional functions.
            pass

    # Basis Function Calls

    def _indicator(self, x):
        phi = mdp.numx.zeros([x.shape[0], self._output_dim])
        phi[range(x.shape[0]), x[:, 0].astype('int')] = 1
        return phi

    def _fourier(self, x):
        x = self._normalize(x)
        return mdp.numx.asarray([mdp.numx.cos(mdp.numx.pi * mult(self._coeffs, _x)) for _x in x])

    def _polynomial(self, x):
        x = self._normalize(x)
        return mdp.numx.asarray([[mdp.numx.prod(map(pow, _x, p)) for p in self._powers] for _x in x])

    def _radial(self, x):
        return self._rbfnode(x)

    def _lem(self, x):
        return self.v[x.ravel().astype('int'), :]

    def _gsfa(self, x):
        for layernum in xrange(self._nlayers):
            z = self.fa[layernum](x)
            if layernum == self._nlayers - 1:
                x = mult(z, self.v[layernum][:, 1:self.output_dim + 1])
            else:
                x = mult(z, self.v[layernum][:, :self._npoly])
        return x

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def is_invertible():
        return False

    # internal utility methods

    def _normalize(self, x):
        _range = self.lims[1, :] - self.lims[0, :]
        return mdp.numx.asarray([(x[:, i] - self.lims[0, i]) / _range[i] if _range[i]
                                 else 0.0 for i in xrange(self._input_dim)]).T

    @staticmethod
    def _get_laplacian(adj, normalize='True'):
        if normalize:
            d = adj.sum(axis=1)
            identity = mdp.numx.identity(len(d))
            mat_lapl = identity * d - adj
            osd = mdp.numx.zeros(len(d))
            for i in range(len(d)):
                if d[i] > 0:
                    osd[i] = mdp.numx.sqrt(1.0 / d[i])
            t = identity * osd
            return mult(t, mult(mat_lapl, t))
        else:
            mat_degree = mdp.numx.diag(adj.sum(axis=0))
            return mat_degree - adj

    @staticmethod
    def _get_neighbors(adj, x):
        return mdp.numx.argwhere(adj[x]).ravel()

    def _train(self, x):
        pass

    def _execute(self, x):
        _fn = getattr(self, '_%s' % self.basis_name)
        return self._refcast(_fn(self._gp(x)) * self.scale_out)

    @staticmethod
    def _scale_to_unit_interval(ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    def _tile_raster_images(self, x, img_shape, tile_shape, tile_spacing=(0, 0), scale_rows_to_unit_interval=True,
                            output_pixel_vals=True):
        """
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.
        (Borrowed from Theano)

        This function is useful for visualizing datasets whose rows are images,
        and also columns of matrices for transforming those rows
        (such as the first layer of a neural net).

        :type x: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
        :param x: a 2-D array in which every row is a flattened image.

        :type img_shape: tuple; (height, width)
        :param img_shape: the original shape of each image

        :type tile_shape: tuple; (rows, cols)
        :param tile_shape: the number of images to tile (rows, cols)

        :param output_pixel_vals: if output should be pixel values (i.e. int8
        values) or floats

        :param scale_rows_to_unit_interval: if the values need to be scaled before
        being plotted to [0,1] or not

        :returns: array suitable for viewing as an image.
        (See:`PIL.Image.fromarray`.)
        :rtype: a 2-d array with same dtype as x.

        """

        assert (len(img_shape) == 2)
        assert (len(tile_shape) == 2)
        assert (len(tile_spacing) == 2)

        # The expression below can be re-written in a more C style as
        # follows :
        #
        # out_shape = [0,0]
        # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
        #                tile_spacing[0]
        # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
        #                tile_spacing[1]
        out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                     in zip(img_shape, tile_shape, tile_spacing)]

        if isinstance(x, tuple):
            assert len(x) == 4
            # Create an output ndarray to store the image
            if output_pixel_vals:
                out_array = mdp.numx.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
            else:
                out_array = mdp.numx.zeros((out_shape[0], out_shape[1], 4))

            # colors default to 0, alpha defaults to 1 (opaque)
            if output_pixel_vals:
                channel_defaults = [0, 0, 0, 255]
            else:
                channel_defaults = [0., 0., 0., 1.]

            for i in xrange(4):
                if x[i] is None:
                    # if channel is None, fill it with zeros of the correct
                    # dtype
                    out_array[:, :, i] = mdp.numx.zeros(out_shape,
                                                        dtype='uint8' if output_pixel_vals else out_array.dtype) + \
                                         channel_defaults[i]
                else:
                    # use a recurrent call to compute the channel and store it
                    # in the output
                    out_array[:, :, i] = self._tile_raster_images(x[i], img_shape, tile_shape, tile_spacing,
                                                                  scale_rows_to_unit_interval, output_pixel_vals)
            return out_array

        else:
            # if we are dealing with only one channel
            h, w = img_shape
            hs, ws = tile_spacing

            # generate a matrix to store the output
            if output_pixel_vals:
                out_array = 155 * mdp.numx.ones(out_shape, dtype='uint8')
            else:
                out_array = mdp.numx.zeros(out_shape, dtype=x.dtype)

            for tile_row in xrange(tile_shape[0]):
                for tile_col in xrange(tile_shape[1]):
                    if tile_row * tile_shape[1] + tile_col < x.shape[0]:
                        if scale_rows_to_unit_interval:
                            # if we should scale values to be between 0 and 1
                            # do this by calling the `scale_to_unit_interval`
                            # function
                            this_img = self._scale_to_unit_interval(
                                x[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                        else:
                            this_img = x[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                        # add the slice to the corresponding position in the
                        # output array
                        out_array[tile_row * (h + hs):tile_row * (h + hs) + h,
                        tile_col * (w + ws):tile_col * (w + ws) + w] = this_img * (255 if output_pixel_vals else 1)
            return out_array

    @staticmethod
    def _matplotlib_fig_arr(fig):
        """
        Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it.
        fig - a matplotlib figure
        """
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError:
            return None

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = mdp.numx.fromstring(canvas.tostring_rgb(), dtype='uint8')
        buf.shape = (h, w, 3)

        return buf

    def get_fn_responses_img(self):
        """returns an image of features sampled from the environment"""
        if self.input_dim == 1:
            try:
                from matplotlib import pyplot as plt
            except ImportError:
                return None

            if self._basis_mode == 'discrete':
                x = mdp.numx.atleast_2d(mdp.numx.arange(self.lims[0, 0], self.lims[1, 0])).T
            else:
                x = mdp.numx.atleast_2d(mdp.numx.linspace(self.lims[0, 0], self.lims[1, 0])).T
            y = self._execute(x.astype('float'))
            fig = plt.figure(facecolor='black', edgecolor='g')
            _tmp = mdp.numx.ceil(mdp.numx.sqrt(y.shape[1]))
            for i in xrange(y.shape[1]):
                ax = fig.add_subplot(_tmp, _tmp, i + 1, axisbg='k')
                ax.plot(x, y[:, i], lw=3, c='y')
                for axis in ['top', 'bottom', 'left', 'right']:
                    ax.spines[axis].set_linewidth(3)
                    ax.spines[axis].set_color('w')
                    # ax.tick_params(axis='x', colors='w')
                    # ax.tick_params(axis='y', colors='w')
            plt.tight_layout(pad=0.0)
            img = self._matplotlib_fig_arr(fig)
            plt.close()
            return img
        elif self.input_dim == 2:
            if self._basis_mode == 'discrete':
                _x, _y = mdp.numx.meshgrid(mdp.numx.linspace(self.lims[0, 0], self.lims[1, 0],
                                                             self.lims[1, 0] + 1, endpoint=True),
                                           mdp.numx.linspace(self.lims[0, 1], self.lims[1, 1],
                                                             self.lims[1, 1] + 1, endpoint=True))
            else:
                _x, _y = mdp.numx.meshgrid(mdp.numx.linspace(self.lims[0, 0], self.lims[1, 0], 20, endpoint=True),
                                           mdp.numx.linspace(self.lims[0, 1], self.lims[1, 1], 20, endpoint=True))
            x = mdp.numx.asarray(zip(_x.ravel(), _y.ravel()))
            z = self._execute(x.astype('float'))
            _tmp1 = _tmp2 = int(mdp.numx.ceil(mdp.numx.sqrt(z.shape[1])))
            if z.shape[1] <= _tmp1 * (_tmp1 - 1):
                _tmp2 = _tmp1 - 1
            return self._tile_raster_images(x=z.T, img_shape=(_x.shape[0], _x.shape[1]), tile_shape=(_tmp1, _tmp2),
                                            tile_spacing=(1, 1), output_pixel_vals=False)
        return None

    def __repr__(self):
        name = type(self).__name__
        basis_name = "basis_name=%s" % str(self.basis_name)
        lims = "lims=%s" % str([list(self.lims[0]), list(self.lims[1])])
        order = "order=%s" % str(self.order)
        decoupled = "decoupled=%s" % str(self.decoupled)
        basis_params = "basis_params=%s" % str(self.basis_params)
        scale_out = "scale_out='%s'" % str(self.scale_out)
        n_grid_pts = "n_grid_pts=%s" % str(self._gp.n_grid_pts)
        input_dim = "input_dim=%s" % str(self.input_dim)
        dtype = "dtype=%s" % str(self.dtype)
        args = ', '.join((basis_name, lims, order, decoupled, basis_params, scale_out, n_grid_pts, input_dim, dtype))
        return name + '(' + args + ')'
