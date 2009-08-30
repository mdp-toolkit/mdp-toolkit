import mdp
from mdp import numx

import shogun.Kernel as sgKernel
import shogun.Features as sgFeatures
import shogun.Classifier as sgClassifier

class _SVMNode(mdp.Node):
    def is_invertible(self):
        return False
    
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n
    
    def _set_output_dim(self, n):
        msg = "Output dim cannot be set explicitly!"
        raise mdp.NodeException(msg)

class _OrderedDict:
    """Very simple version of an ordered dict."""
    def __init__(self, items):
        self._keys = []
        self._vals = []
        self.update(items)
    def update(self, other):
        """Update an ordered dict with new values."""
        for entry in other:
            if isinstance(other, dict):
                new_key = entry
                new_val = other[entry]
            else:
                new_key = entry[0]
                new_val = entry[1]
            if new_key in self._keys:
                i = self._keys.index(new_key)
                self._vals[i] = new_val
            else:
                self._keys.append(new_key)
                self._vals.append(new_val)

class ShogunSVMNode(_SVMNode):
    """The ShogunSVMNode works as a wrapper class for accessing the shogun library
    for support vector machines.
    """
    default_parameters = {
        'C': 1,
        'epsilon': 1e-3,
    }
    
    # Swig-code does not work with named parameters, so we have to define an order
    kernel_parameters = {
        'PolyKernel': [('size',10), ('degree',3), ('inhomogene',True)],
        'GaussianKernel': [('size',10), ('width',1)],
        'LinearKernel': [],
        'SigmoidKernel': [('size',10), ('gamma',1), ('coef0', 0)]
    }
    
    def __init__(self, svm_library="shogun", classifier="libsvmmulticlass", classifier_options=None,
                 kernel="GaussianKernel", kernel_options=None,
                 num_threads="autodetect", input_dim=None, dtype=None):
        """
        Keyword arguments:
            
            svm_library -- The type of the SVM library, "libsvm" or "shogun"
            classifier  -- The classifier to use
            classifier_options -- Options for the classifier
            kernel      -- The kernel to use. Default parameters are specified for
                             "PolyKernel"
                             "GaussianKernel"
                             "LinearKernel"
                             "SigmoidKernel"
                            Further kernels are possible if they are included in shogun
                            and if kernel_options provides the correct init arguments.
            kernel_options -- For known kernels, a dict specifying the options is possible,
                           options not included take a default value.
                           Unknown kernels need an ordered list of constructor arguments.
            num_threads -- The number of threads, shogun should use
                           can be set to "autodetect", then shogun will use the number of cpu cores.
                           Attention: this could crash on windows
        
        """
        if classifier_options is None:
            classifier_options = {}
        
        self._svm_library = svm_library
        self._num_threads = num_threads
        self._x = numx.array([])
        self._cl = numx.array([])
        self._norm_labels = numx.array([])
        self._label_map = {}
        self._classification_type = "multi"
        
        self.set_classifier(classifier)
        
        self.classifier_options = self.default_parameters
        self.classifier_options.update(classifier_options)
        
        for p in self.classifier_options.keys():
            self.set_classifier_param(p, self.classifier_options[p])

        self.set_kernel(kernel, kernel_options)
        
        super(ShogunSVMNode, self).__init__(input_dim=input_dim,
                                       output_dim=None, dtype=dtype)

    def set_classifier(self, name="libsvm"):
        """Sets and initialises the classifier. If a classifier is reset by the user, 
        the parameters will have to be set again.

        'name' can be a string, a subclass of shogun.Classifier or an instance of such
        a class.
        """
        self._classifier = None
        self.svm = None
        
        if isinstance(name, sgClassifier.Classifier):
            self._classifier = name.__class__
            self.svm = name
        
        if isinstance(name, basestring):
            #FIXME: this does not work in python2.4
            #XXX why adding None to the list?
            possibleNames = [name if name in dir(sgClassifier) else None] + \
                            [s for s in dir(sgClassifier) if s.lower()==name.lower()]
            for s in possibleNames:
                #XXX why do we need this try/except if we already found the name?
                try:
                    self._classifier = getattr(sgClassifier, s)
                    break
                except Exception:
                    pass
            #XXX why another try/except?
            #FIXME: couldn't we just check that len(possibleNames) > 0?
            try:
                self.svm = self._classifier()
            except TypeError:
                msg = "Library '%s' is not known." % name
                raise mdp.NodeException(msg)

        #XXX: could you comment on that?
        if self._classifier is not None and self.svm is None:
            if issubclass(name, sgClassifier.Classifier):
                self.svm = self._classifier()

        #XXX: could you comment on that?
        if self._classifier is None:
            if issubclass(name, sgClassifier.Classifier):
                self._classifier = name
                self.svm = self._classifier()

        #XXX: could you comment on that?
        if self.svm is None:
            msg = "The classifier '%s' is not supported." %name
            raise mdp.NodeException(msg)

        #XXX: could you comment on that?
        if not issubclass(self._classifier, sgClassifier.Classifier):
            msg = "The classifier '%s' is no subclass of CClassifier." % self._classifier.__name__
            raise mdp.NodeException(msg)

        self._set_num_threads()
        self._classification_type = self._get_classification_type()

    def _get_classification_type(self):
        duals = ["LibSVM", "SVMLin"]
        if self._classifier.__name__ in duals:
            return "dual"
        else:
            return "multi"

    def _set_num_threads(self):
        # init number of threads
        if self._num_threads == "autodetect":
            try:
                self._num_threads = self.svm.parallel.get_num_cpus()
            except SystemError:
                # We're helping shogun here
                self._num_threads = 1
        self.svm.parallel.set_num_threads(self._num_threads)

    def set_classifier_param(self, param, *value):
        """Sets parameters for the classifier.
        """
        # Non-standard cases
        if param == "C" and len(value) == 1:
            value += value
        # get the parameter setting method
        meth = getattr(self.svm, "set_"+param)
        # call it
        meth(*value)

    def set_kernel(self, name, options=None):
        """Sets the Kernel along with options.
        """
        if options is None:
            options = {}
        if name in ShogunSVMNode.kernel_parameters and not isinstance(options, list):
            default_opts = _OrderedDict(ShogunSVMNode.kernel_parameters[name])
            default_opts.update(options)
            #FIXME: do we need a print here?
            print default_opts._vals
            self.kernel = getattr(sgKernel, name)(*(default_opts._vals))
        else:
            self.kernel = getattr(sgKernel, name)(*options)

    def _normalize_labels(self, mode=None):
        """To avoid problems with the algorithms, we normalise the labels to a standard layout
        and take care of the mapping
        """
        if mode == None:
            mode = self._classification_type
        labels = set(self._cl)
        if mode == "dual":
            if len(labels) > 2:
                msg = "In dual mode only two labels can be given"
                raise mdp.NodeException(msg)
            # pop first label and reduce
            if len(labels) > 0:
                l = labels.pop()
                self._label_map[-1] = l
            if len(labels) > 0:
                l = labels.pop()
                self._label_map[1] = l
            else:
                msg = "Training your SVM with only one label is not the most sensible thing to do."
                raise mdp.MDPWarning(msg)
        elif mode == "multi":
            count = 0
            for l in labels:
                self._label_map[count] = l
                count += 1
        else:
            msg = "Remapping mode not known"
            raise mdp.NodeException(msg)

        # now execute the mapping
        try:
            inverted = dict([(v, k) for k, v in self._label_map.iteritems()])
        except TypeError:
            # put more elaborated code here for circumventing this issue
            msg = "Problem inverting. Labels maybe not hashable."
            raise mdp.NodeException(msg)
        norm_labels = map(inverted.get, self._cl)
        
        if None in self._norm_labels:
            msg = "Error in remapping method"
            raise mdp.NodeException(msg)
            
        self._norm_labels = mdp.numx.array(norm_labels)
        
    def _check_train_args(self, x, cl):
        if (isinstance(cl, (list, tuple, numx.ndarray)) and
            len(cl) != x.shape[0]):
            msg = ("The number of labels should be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)
    
    def _reshape_data(self):
        x_size = self._x.size
        #FIXME: what if size is odd?
        #XXX: why this reshape? 
        self._x = self._x.reshape((x_size//2, 2))
    
    def _stop_training(self):
        self._reshape_data()
        self._normalize_labels()
        
        self.features = sgFeatures.RealFeatures(self._x.transpose())
        
        if issubclass(self._classifier, sgClassifier.LinearClassifier):
            self.svm.set_features(self.features)
        else:
            self.kernel.init(self.features, self.features)
            self.svm.set_kernel(self.kernel)
        # shogun expects floats
        labels = sgFeatures.Labels(self._norm_labels.astype(float))
        self.svm.set_labels(labels)
        
        self.svm.train()
        
    def _train(self, x, cl):
        """Update the internal structures according to the input data 'x'.
        
        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """
        self._x = mdp.numx.append( self._x, x )
        # if cl is a number, all x's belong to the same class
        if isinstance(cl, (list, tuple, numx.ndarray)):
            self._cl = mdp.numx.append( self._cl, cl )
        else:
            cls = [cl] * len(x)
            self._cl = mdp.numx.append( self._cl, cls )

    def training_set(self, ordered=False):
        """Shows the set of data that has been inserted to be trained."""
        self._reshape_data()
        if ordered:
            z = zip(self._cl, self._x)
            data = {}
            for k, v in z:
                if data.has_key(k):
                    data[k] = data[k] + (v,)
                else:
                    data[k] = (v,)
            return data
        else:
            return zip(self._cl, self._x)
        
    def classify(self, x):
        """Classify the input data 'x'
        """
        self._pre_execution_checks(x)
        
        test = sgFeatures.RealFeatures(x.transpose())
        if issubclass(self._classifier, sgClassifier.LinearClassifier):
            self.svm.set_features(self.features)
        else:
            self.kernel.init(self.features, test)

#       still some problems with the backmapping 
        if self._classification_type == "dual":
            return self.svm.classify().get_labels()
        else:
            labels = map(self._label_map.get, self.svm.classify().get_labels())
            return labels


