from mdp import numx, numx_linalg, utils, NodeException
import mdp
import scipy.signal as signal

# TODO dependency on scipy.signal

# TODO mode is read-only

# TODO FFT convolution

# TODO automatic selection of convolution

# TODO provide generators for standard filters

class Convolution2DNode(mdp.Node):
    def __init__(self, filters, input_shape = None,
                 mode = 'full', boundary = 'fill', fillvalue = 0,
                 output_2d = True,
                 input_dim = None, dtype = None):
        """
        Input arguments:

        mode -- convolution mode, as defined in scipy.signal.convolve2d
                'mode' is one of ['valid', 'same', 'full']
                Default is 'full'
        boundary -- 'boundary' is one of ['fill', 'wrap', 'symm']
        output_2d -- If True, the output array is 2D; the first index
                     corresponds to data points; every output data point
                     is the result of flattened convolution results, with
                     the output of each filter concatenated together.
                     
                     If False, the output array is 4D; the format is
                     data[idx,filter_nr,x,y]
                     filter_nr: index of convolution filter
                     idx: data point index
                     x, y: 2D coordinates
        """
        super(Convolution2DNode, self).__init__(input_dim=input_dim,
                                              dtype=dtype)
        # TODO: check dtype of filters, 2D shape
        self.filters = filters
        self.input_shape = input_shape
        self.mode = mode
        self.boundary = boundary
        self.fillvalue = fillvalue
        self.output_2d = output_2d
        self.output_shape = None

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _pre_execution_checks(self, x):
        """This method contains all pre-execution checks.
        It can be used when a subclass defines multiple execution methods.

        In this case, the output dimension depends on the type of
        convolution we use (padding, full, ...). Also, we want to
        to be able to accept 3D arrays.
        """

        self._check_input(x)
        
        # TODO set output_dim automatically

    def _check_input(self, x):
        # check input rank
        if not x.ndim in [2,3]:
            error_str = "x has rank %d, should be 2 or 3" % (x.ndim)
            raise NodeException(error_str)

        # set 2D shape if necessary
        if self.input_shape is None:
            if x.ndim == 2:
                error_str = "Cannot infer 2D shape from 1D data points. " + \
                            "Data must have rank 3, or shape argument given."
                raise NodeException(error_str)
            else:
                self.input_shape = x.shape[1:]

        # set the input dimension if necessary
        if self.input_dim is None:
            self.input_dim = numx.prod(self.input_shape)

        # set the dtype if necessary
        if self.dtype is None:
            self.dtype = x.dtype

        # check the input dimension
        if not numx.prod(x.shape[1:]) == self.input_dim:
            error_str = "x has dimension %d, should be %d" % (x.shape[1],
                                                              self.input_dim)
            raise NodeException(error_str)

        # set output_dim if necessary
        if self.output_dim is None:
            input_shape = self.input_shape
            filters_shape = self.filters.shape
            if self.mode == 'same':
                self.output_shape = input_shape
            elif self.mode == 'full':
                self.output_shape = (input_shape[0]+filters_shape[1]-1,
                                     input_shape[1]+filters_shape[2]-1)
            else: # mode == 'valid'
                self.output_shape = (input_shape[0]-filters_shape[1]+1,
                                     input_shape[1]-filters_shape[2]+1)
            self.output_dim = self.filters.shape[0]*numx.prod(self.output_shape)

        if x.shape[0] == 0:
            error_str = "x must have at least one observation (zero given)"
            raise NodeException(error_str)


    def _execute(self, x):
        is_2d = x.ndim==2
        output_shape, input_shape = self.output_shape, self.input_shape
        filters = self.filters
        nfilters = filters.shape[0]

        # XXX depends on convolution
        y = numx.empty((x.shape[0], nfilters,
                        output_shape[0], output_shape[1]), dtype=self.dtype)
        for n_im, im in enumerate(x):
            if is_2d:
                im = im.reshape(input_shape)
            for n_flt, flt in enumerate(filters):
                y[n_im,n_flt,:,:] = signal.convolve2d(im, flt,
                                                      mode=self.mode,
                                                      boundary=self.boundary,
                                                      fillvalue=self.fillvalue)
                
        # reshape if necessary
        if self.output_2d:
            y.resize((y.shape[0], self.output_dim))
        
        return y
    
if __name__=='__main__':
    import numpy, mdp
    im = numpy.random.rand(4, 3,3)
    node = mdp.nodes.Convolution2DNode(numpy.array([[[1.]]]))
    node.execute(im)
    
