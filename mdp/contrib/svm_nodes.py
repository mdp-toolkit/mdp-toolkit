import mdp
from mdp import numx

class _SVMNode(mdp.Node):
    def is_invertible(self):
        return False
    
    def _set_input_dim(self, n):
        self._input_dim = n
        self._output_dim = n
    
    def _set_output_dim(self, n):
        msg = "Output dim cannot be set explicitly!"
        raise mdp.NodeException(msg)

class ShogunSVMNode(_SVMNode):
    # must be included when I know what parameters are reasonable
    default_parameters = {
        'C': 1,
        'epsilon': 1e-3,
    }
    import shogun.Kernel as Kernel
    import shogun.Features as Features
    import shogun.Classifier as Classifier
    def __init__(self, svm_library="libsvm", classifier="libsvmmulti", classifier_options=(),
                 kernel="GaussianKernel", kernel_options=(),
                 num_threads=1, input_dim=None, dtype=None):
        """
        Keyword arguments:
            
            svm_library -- The type of the SVM library, "libsvm" or "shogun"
            num_threads -- The number of threads, shogun should use
                           can be set to "auto", then shogun will use the number of cpu cores.
                           Attention: this will crash on windows (but it does not say so in shogun's docs)
        
        """
#        if svm_library == "libsvm":
#            import svm
#        elif svm_library == "shogun":
#            from shogun.Classifier import SVM
#        else:
#            msg = "SVM library '%s' is not supported!" % svm_library
#            raise mdp.NodeException(msg)

        self._svm_library = svm_library
        self._num_threads = num_threads
        self._x = numx.array([])
        self._cl = numx.array([])
        self._norm_labels = numx.array([])
        self.label_map = {}
        
        self.set_classifier(classifier)
        
        C=0.017
        epsilon=1e-5
        self.set_classifier_param("C", C, C)
        self.set_classifier_param("epsilon", epsilon)
        # TODO: user input...
        
        kernel_cache = 10
        width=2.1
        self.set_kernel("GaussianKernel", kernel_cache, width)
        
        super(ShogunSVMNode, self).__init__(input_dim=input_dim,
                                       output_dim=None, dtype=dtype)

    def set_classifier(self, name="libsvm"):
        """Sets and initialises the classifier. If a classifier is reset by the user, 
        the parameters will have to be set again.
        """
        if isinstance(name, basestring):
            if name == "libsvm":
                self._classifier = self.Classifier.LibSVM
            elif name == "libsvmmulti":
                self._classifier = self.Classifier.LibSVMMultiClass
            else:
                msg = "Library '%s' not known" % name
                raise mdp.NodeException(msg)
        elif isinstance(name, type):
            self._classifier = classifier
        else:
            msg = "Type not supported"
            raise mdp.NodeException(msg)
        self.svm=self._classifier()
        # init number of threads
        if self._num_threads == "auto":
            try:
                self._num_threads = self.svm.parallel.get_num_cpus()
            except:
                # We're helping shogun here
                self._num_threads = 1
        self.svm.parallel.set_num_threads(self._num_threads)
    
    def set_classifier_param(self, param, *value):
        """Sets parameters for the classifier.
        """
        getattr(self.svm, "set_"+param)(*value)

    def set_kernel(self, name, *options):
        """Sets the Kernel along with options.
        """
        self.kernel = getattr(self.Kernel, name)(*options)

    def _normalize_labels(self, mode):
        """To avoid problems with the algorithms, we normalise the labels to a standard layout
        and take care of the mapping
        """
        labels = set(self._cl)
        if mode == "dual":
            if len(labels) > 2:
                msg = "In dual mode only two labels can be given"
                raise mdp.NodeException(msg)
            # pop first label and reduce
            if len(labels) > 0:
                l = labels.pop()
                self.label_map[-1] = l
            if len(labels) > 0:
                l = labels.pop()
                self.label_map[1] = l
            else:
                msg = "Training your SVM with only one label is not the most sensible thing to do."
                raise mdp.MDPWarning(msg)
        elif mode == "multi":
            count = 0
            for l in labels:
                self.label_map[count] = l
                count += 1
        else:
            msg = "Remapping mode not known"
            raise mdp.NodeException(msg)

        # now execute the mapping
        try:
            inverted = dict([(v, k) for k, v in self.label_map.iteritems()])
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
        l = self._x.size
        self._x = self._x.reshape( (l/2,2) )
    
    def _stop_training(self):
        self._reshape_data()
        if self._classifier == self.Classifier.LibSVM:
            self._normalize_labels("dual")
        else:
            self._normalize_labels("multi")

        self.features = self.Features.RealFeatures(self._x.transpose())
        self.kernel.init(self.features, self.features)
        self.svm.set_kernel(self.kernel)
        # shogun expects floats
        labels = self.Features.Labels(self._norm_labels.astype(float))
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
            for val in z:
                k = val[0]
                v = val[1]
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
        
        test = self.Features.RealFeatures(x.transpose())
        self.kernel.init(self.features, test)

#       still problems with the backmapping 
#        labels = map(self.label_map.get, self.svm.classify().get_labels())

        return self.svm.classify().get_labels()


class LibSVMNode(_SVMNode):
    import svm
