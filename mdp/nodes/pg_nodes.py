import mdp
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from multiprocessing import Process, Queue


class PG2DNode(mdp.PreserveDimNode):
    """ PG2DNode is a non-blocking fast online 2D plotting node. It uses
    the fast plotting python library called the PyQtGraph (http://www.pyqtgraph.org/).

    PG2DNode is a non-trainable node. It works similar to an oscilloscope;
    place it between two OnlineNodes in an OnlineFlow to visualize the data being
    passed between the nodes. The node can also be used standalone, that is it plots
    as soon as new data is passed through the execute call.

    PG2DNode 'subclasses' should take care of overwriting these functions
    '_setup_plots' - setup your plots, layout, etc.
    '_update_plots' - data updates to the plots.
    Check PGCurveNode and PGImageNode as examples and also the examples provided within the PyQtGraph package.

    Care must be taken to not overwrite methods '_check_input', '__pg_process', '__pg_data' and
    '__plot' in a subclass. These are responsible for spawning a parallel plot process.

    PG2DNode also works like an identity node returning the input as the output.

    When the plotting windows are manually closed, the node continues to transmit input
    as the output without interfering the flow.
    """

    def __init__(self, use_buffer=False, x_range=None, y_range=None, interval=1, timeout=0, window_title=None,
                 window_size_xy=(640, 480), input_dim=None, output_dim=None, dtype=None):
        """
        user_buffer: If the data arrives sample by sample (like in an OnlineFlow), use_buffer can be set to store
        samples in a circular buffer. At each time-step the buffer contents are displayed.

        x_range: Denotes the range of x-axis values to be shown. When the use_buffer is set,
                this also denotes the size of the buffer.

        y_range: y-axis range

        interval: Number of execute calls after which the plots are updated.
                 1 - Plots are updated after each execute call
                 10 - Plots are updated after every 10th execute call
                 -1 - Automatically optimize the interval such that the plot updates do not
                      slow the flow's execution time.

        timeout: Sets a minimum of timeout msecs for each plot update. Default is 0.

        window_title: Window title

        window_size: XY size tuple of the window

         """
        super(PG2DNode, self).__init__(input_dim, output_dim, dtype)
        self.use_buffer = use_buffer
        self._x_range = x_range
        self._y_range = y_range
        self._interval = 1 if interval == -1 else interval
        self._given_interval = interval
        self._timeout = timeout
        self._window_title = window_title
        self._window_size_xy = window_size_xy

        if use_buffer:
            if x_range is None:
                raise mdp.NodeException("Provide x_range to init buffer size.")
            self._buffer = mdp.nodes.NumxBufferNode(buffer_size=x_range[1])
        self._flow_time = 0
        self._tlen = 0

        # child process that is set upon execution
        self._viewer = None
        # queue to communicate data between the main and the child process
        self.new_data = Queue(1)

        # all these variables are set in the child process
        self._win = None  # pyqtgraph window
        self._layout = None  # pyqtgraph plots layout
        self._plot_items = None  # pyqtgraph plot items
        self._plot_objects = None  # pyqtgraph plot objects (graphics objects)

    # properties

    def get_x_range(self):
        """Returns the range for x axis."""
        return self._x_range

    def set_x_range(self, x_range):
        """Sets the range for x axis."""
        if x_range is None:
            return
        if (not isinstance(x_range, tuple)) and (not isinstance(x_range, list)):
            raise mdp.NodeException("x_range must be a tuple or a list and not %s." % str(type(x_range)))
        if len(x_range) != 2:
            raise mdp.NodeException("x_range must contain 2 elements, given %s." % len(x_range))
        self._x_range = x_range

    x_range = property(get_x_range, set_x_range, doc="x-axis range")

    def get_y_range(self):
        """Returns the range for y axis."""
        return self._y_range

    def set_y_range(self, y_range):
        """Sets the range for y axis."""
        if y_range is None:
            return
        if (not isinstance(y_range, tuple)) and (not isinstance(y_range, list)):
            raise mdp.NodeException("x_range must be a tuple or a list and not %s." % str(type(y_range)))
        if len(y_range) != 2:
            raise mdp.NodeException("x_range must contain 2 elements, given %s." % len(y_range))
        self._y_range = y_range

    y_range = property(get_y_range, set_y_range, doc="y-axis range")

    @staticmethod
    def _get_pglut(lutname=None):
        """Colorspace look up table (requires Matplotlib)"""
        pg_lut = None
        if lutname is not None:
            try:
                from matplotlib.cm import get_cmap
                from matplotlib.colors import ColorConverter
            except ImportError:
                return None
            lut = []
            cmap = get_cmap(lutname, 1000)
            for i in range(1000):
                r, g, b = ColorConverter().to_rgb(cmap(i))
                lut.append([r * 255, g * 255, b * 255])
            pg_lut = mdp.numx.array(lut, dtype=mdp.numx.uint8)
            pg_lut[0, :] = [0, 0, 0]
        return pg_lut

    @staticmethod
    def is_trainable():
        return False

    def _get_supported_dtypes(self):
        return mdp.utils.get_dtypes('AllInteger') + mdp.utils.get_dtypes('Float')

    # -------------------------------------------
    # super private methods.
    # Do not overwrite unless you know what you are doing.

    def _check_input(self, x):
        super(PG2DNode, self)._check_input(x)
        if self._viewer is None:
            self._viewer = Process(target=self.__pg_process)
            self._viewer.start()

    def __pg_process(self):
        # spawned process
        self.app = QtGui.QApplication([])
        # create and display a plotting window
        self._win = pg.GraphicsWindow()
        if self._window_title is None:
            self._win.setWindowTitle(type(self).__name__)
        else:
            self._win.setWindowTitle(self._window_title)
        self._win.resize(*self._window_size_xy)
        # get the layout to be displayed in this window
        self._plot_objects, self._plot_items, self._layout = self._setup_plots()
        # Set the layout as a central item
        self._win.setCentralItem(self._layout)
        self._win.show()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.__pg_data)
        self.timer.start(self._timeout)
        self.app.exec_()

    def __pg_data(self):
        # communication function
        if not self.new_data.empty():
            data = self.new_data.get()
            if data is None:
                QtGui.QApplication.closeAllWindows()
                return
            self._update_plots(data)

    def __plot(self, x):
        # plot given data
        if not self._viewer.is_alive():
            return
        while self.new_data.full():
            if not self._viewer.is_alive():
                self.new_data.get()  # empty Queue
                return
            time.sleep(0.0001)
        self.new_data.put(x)

    # -------------------------------------------

    def _setup_plots(self):
        # Setup your plots, layout, etc. Overwrite in the subclass
        # Check PyQtGraph for examples
        # run in shell prompt: python -c "import pyqtgraph.examples; pyqtgraph.examples.run()"
        self._win.setWindowTitle("Uninitialized Blank Plot")

    def _update_plots(self, x):
        # Update your individual plotitems. Overwrite in the subclass
        pass

    def _execute(self, x):
        self._tlen += 1
        _flow_dur = time.time() - self._flow_time
        y = x
        if self.use_buffer:
            self._buffer.train(x)
            y = self._buffer(x)
        if self._tlen % int(self._interval) == 0:
            t = time.time()
            self.__plot(y)
            _plot_dur = time.time() - t
            if self._given_interval == -1:
                self._interval = self._interval * (100 * _plot_dur / _flow_dur +
                                                   (self._tlen / self._interval - 1) * self._interval) / float(
                    self._tlen)
                self._interval = mdp.numx.clip(self._interval, 1, 50)
        self._flow_time = time.time()
        return x

    def stop_rendering(self):
        # close all the plots.
        if self._viewer is not None:
            self.__plot(None)
            self._viewer.join()


class PGCurveNode(PG2DNode):
    """ PGCurveNode is a PG2DNode that displays the input data as multiple curves.
        use_buffer needs to be set if the data arrives sample by sample.
    """

    def __init__(self, display_dims=None, split_figs=False, plot_titles=None, use_buffer=False, x_range=None,
                 y_range=None, interval=1, timeout=0, window_title=None, window_size_xy=(640, 480), input_dim=None,
                 output_dim=None, dtype=None):
        """
        display_dims: Dimensions that are displayed in the plots. By default all dimensions are displayed.
                      Accepted values: scalar/list/array - displays the provided dimensions

        split_figs: When set, each data dimension is plotted in a separate figure, otherwise they are vertically stacked
        in a single plot.

        plot_titles: A string or a list of title strings for each plot if split_figs is set to True.

        """
        super(PGCurveNode, self).__init__(use_buffer=use_buffer, x_range=x_range, y_range=y_range, interval=interval,
                                          timeout=timeout, window_title=window_title, window_size_xy=window_size_xy,
                                          input_dim=input_dim, output_dim=output_dim, dtype=dtype)

        if display_dims is not None:
            if mdp.numx.isscalar(display_dims):
                display_dims = [display_dims]
            display_dims = mdp.numx.asarray(display_dims)
        self.display_dims = display_dims

        self._split_figs = split_figs
        self._plot_titles = [] if plot_titles is None else plot_titles

    def _set_input_dim(self, n):
        self._input_dim = n
        if self.display_dims is None:
            self.display_dims = range(n)

    def _setup_plots(self):
        layout = pg.GraphicsLayout()
        n_disp_dims = len(self.display_dims)
        plot_items = [pg.PlotItem() for _ in xrange(n_disp_dims)]
        curves = [pg.PlotCurveItem(pen=(i, n_disp_dims * 1.3)) for i in xrange(n_disp_dims)]
        plot_objects = {'curves': curves}
        if self._split_figs:
            num_rows = mdp.numx.ceil(mdp.numx.sqrt(n_disp_dims))
            for i in xrange(n_disp_dims):
                plot_items[i].addItem(curves[i])
                layout.addItem(plot_items[i], row=i / num_rows, col=i % num_rows)
                if self.y_range is not None:
                    plot_items[i].setYRange(*self.y_range)
                if self.x_range is not None:
                    plot_items[i].setXRange(*self.x_range)
                if i < len(self._plot_titles):
                    plot_items[i].setTitle(self._plot_titles[i])
        else:
            plot_items = plot_items[0]
            if isinstance(self._plot_titles, (tuple, list)):
                if len(self._plot_titles) > 0:
                    plot_items.setTitle(self._plot_titles[0])
            elif isinstance(self._plot_titles, str):
                plot_items.setTitle(self._plot_titles)
            for i in xrange(n_disp_dims):
                plot_items.addItem(curves[i])
                if n_disp_dims > 1:
                    if self.y_range is None:
                        curves[i].setPos(0, (i + 1) * 6)
                        plot_items.setYRange(0, (n_disp_dims + 1) * 6)
                    else:
                        curves[i].setPos(0, (i + 1) * (self.y_range[1] - self.y_range[0]))
                        plot_items.setYRange(0, (n_disp_dims + 1) * (self.y_range[1] - self.y_range[0]))
                else:
                    if self.y_range is not None:
                        plot_items.setYRange(*self.y_range)

            if self.x_range is not None:
                plot_items.setXRange(*self.x_range)
            layout.addItem(plot_items)

        return plot_objects, plot_items, layout

    def _update_plots(self, x):
        x = x[:, self.display_dims]
        for i in xrange(x.shape[1]):
            self._plot_objects['curves'][i].setData(x[:, i])


class PGImageNode(PG2DNode):
    """ PGImageNode is a PG2DNode that displays the input data as images."""

    def __init__(self, img_shapes, display_dims=None, plot_titles=None, cmap=None, origin='upper',
                 axis_order='row-major', interval=1, timeout=0, window_title=None, window_size_xy=(640, 480),
                 input_dim=None, output_dim=None, dtype=None):
        """
        img_shapes: 2D or 3D shape tuples of each image that is displayed. It is used to reshape the 2D input data.
                   Accepted values: A single tuple - A single image.
                                    A list of tuples - Each shape tuple is used to reshape the corresponding image.
        
        display_dims: Dimensions that are displayed in the plots. By default all dimensions are displayed as a single
                      image plot.
                      Accepted values: list/array - displays the provided dimensions
                                      A list of lists/arrays - display each list/array of dimensions in an
                                                                individual plot

        plot_titles: A string or a list of title strings for each plot

        cmap: Color map to use. Supported: Matplotlib color maps - 'jet', 'gray', etc.

        origin: The origin is set at the upper left hand corner and rows (first dimension of the array)
                are displayed horizontally. It can also be set to 'lower' if you want the first
                row in the array to be at the bottom instead of the top.

        axis_order: Axis order can either be 'row-major' or 'col-major'. For 'row-major', image data is expected
                    in the standard (row, col) order. For 'col-major', image data is expected in
                     reversed (col, row) order.

         """
        super(PGImageNode, self).__init__(use_buffer=False, x_range=None, y_range=None, interval=interval,
                                          timeout=timeout, window_title=window_title, window_size_xy=window_size_xy,
                                          input_dim=input_dim, output_dim=output_dim, dtype=dtype)

        # check img_shapes
        if isinstance(img_shapes, tuple):
            img_shapes = [img_shapes]
        elif isinstance(img_shapes, list):
            for i, img_shape in enumerate(img_shapes):
                if not isinstance(img_shape, tuple):
                    raise mdp.NodeException("'img_shapes' must either be a tuple or a list of tuples. The given "
                                            "list contains %s." % str(type(img_shapes[0])))
        self.img_shapes = img_shapes
        self._n_plots = len(self.img_shapes)

        # check display_dims
        if display_dims is None:
            display_dims = []
            start_dim = 0
            for i in xrange(self._n_plots):
                img_dim = mdp.numx.product(img_shapes[i])
                display_dims += [range(start_dim, start_dim + img_dim)]
                start_dim += img_dim
        if isinstance(display_dims, (list, mdp.numx.ndarray)):
            if not mdp.numx.isscalar(display_dims[0]):
                if len(display_dims) < self._n_plots:
                    raise mdp.NodeException("Length of display dims (%d) does not match with the length of "
                                            "image_shapes (%d)" % (len(display_dims), self._n_plots))

                for i, dim in enumerate(display_dims):
                    if len(dim) != mdp.numx.product(self.img_shapes[i]):
                        raise mdp.NodeException(
                            "Length of %d'th element of 'display_dims' does not match with the "
                            "%d'th 'img_shape' dims (%s)" % (len(dim), i, str(self.img_shapes[i])))
            else:
                display_dims = [display_dims]
                if self._n_plots > 1:
                    raise mdp.NodeException("'display_dims' needs to a list of dims for each plot. Given only "
                                            "dims for one plot.")

                if len(display_dims[0]) != mdp.numx.product(self.img_shapes[0]):
                    raise mdp.NodeException("Length of 'display_dims' (%d) does not match with the "
                                            "'img_shapes' dims (%s)" % (len(display_dims[0]), str(self.img_shapes[0])))
        else:
            raise mdp.NodeException("'display_dims' must be a list/array or a list of lists/arrays "
                                    "and not %s." % str(type(display_dims)))

        self.display_dims = display_dims

        # check plot_titles
        if plot_titles is None:
            plot_titles = []
        if isinstance(plot_titles, str):
            plot_titles = [plot_titles]
        elif isinstance(plot_titles, list):
            for i, title in enumerate(plot_titles):
                if not isinstance(title, str):
                    raise mdp.NodeException("'plot_titles should either be a string or a list of strings. "
                                            " The %d'th element of the given list is of type %s" % (i, type(title)))
        self._plot_titles = plot_titles

        # color map
        self.cmap = cmap

        # origin position
        if origin not in ['upper', 'lower']:
            raise mdp.NodeException("'origin' must either be 'upper' or 'lower' and not %s" % str(origin))
        self.origin = origin

        # axis order
        if axis_order not in ['row-major', 'col-major']:
            raise mdp.NodeException("'axis_order' must either be 'row-major' or 'col-major' "
                                    "and not %s" % str(axis_order))
        self.axis_order = axis_order

    def _setup_plots(self):
        layout = pg.GraphicsLayout()
        plot_items = []
        plot_objects = {'imgs': []}

        num_rows = mdp.numx.ceil(mdp.numx.sqrt(self._n_plots))
        for i in xrange(self._n_plots):
            p = pg.PlotItem()
            img = pg.ImageItem(border='w', lut=self._get_pglut(self.cmap), axisOrder=self.axis_order)
            p.addItem(img)
            # hide axis and set title
            for axis in ['left', 'bottom', 'top', 'right']:
                p.hideAxis(axis)
            if i < len(self._plot_titles):
                p.setTitle(self._plot_titles[i])
            # add to the layout
            layout.addItem(p, row=i / num_rows, col=i % num_rows)
            # store the items
            plot_items.append(p)
            plot_objects['imgs'].append(img)

        return plot_objects, plot_items, layout

    def _update_plots(self, x):
        for i in xrange(self._n_plots):
            img = x[:, self.display_dims[i]].reshape(*self.img_shapes[i])
            if self.origin == "upper":
                if self.axis_order == 'row-major':
                    img = img[::-1]
                elif self.axis_order == 'col-major':
                    img = img[:, ::-1]
            self._plot_objects['imgs'][i].setImage(img)

    def _execute(self, x):
        for i in xrange(x.shape[0]):
            super(PGImageNode, self)._execute(x[i:i + 1])
        return x
