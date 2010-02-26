import mdp
from mdp import numx

import warnings

from svm_nodes import _SVMNode

import shogun.Kernel as sgKernel
import shogun.Features as sgFeatures
import shogun.Classifier as sgClassifier

#Warn wrong version.
#try:
#    version = sgKernel._Kernel.Version_get_version_release()
#except AttributeError:
#    version = ""

# Must check for version 0.8
#
#if not (version.startswith('v0.7') or version.startswith('v0.8')):
#    msg = "Unsupported API version of shogun. Some things may break."
#    warnings.warn(msg, UserWarning)


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
        'PolyKernel': [('size', 10), ('degree', 3), ('inhomogene', True)],
        'GaussianKernel': [('size', 10), ('width', 1)],
        'LinearKernel': [],
        'SigmoidKernel': [('size', 10), ('gamma',1), ('coef0', 0)]
    }

    def __init__(self, classifier="libsvmmulticlass", classifier_options=None,
                 kernel="GaussianKernel", kernel_options=None,
                 num_threads="autodetect", input_dim=None, dtype=None):
        """
        Keyword arguments:
            
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

        self._num_threads = num_threads

        self._classification_type = "multi"

        self.set_classifier(classifier)

        self.classifier_options = self.default_parameters
        self.classifier_options.update(classifier_options)

        for p in self.classifier_options.keys():
            self.set_classifier_param(p, self.classifier_options[p])

        self.set_kernel(kernel, kernel_options)

        super(ShogunSVMNode, self).__init__(input_dim=input_dim, dtype=dtype)

    def set_classifier(self, name="libsvm"):
        """Sets and initialises the classifier. If a classifier is reset by the user, 
        the parameters will have to be set again.
        'name' can be a string, a subclass of shogun.Classifier or an instance of such
        a class
        """

        self._classifier = None
        self.svm = None

        # If name is a classifier instance: Take it
        if not self.svm and isinstance(name, sgClassifier.Classifier):
                self._classifier = name.__class__
                self.svm = name

        # If name is a class: Make an instance
        if not self.svm and name.__class__ == type:
            if issubclass(name, sgClassifier.Classifier):
                self._classifier = name
                self.svm = self._classifier()
            else:
                msg = "The classifier '%s' is no subclass of CClassifier." % self._classifier.__name__
                raise mdp.NodeException(msg)

        # If name is a string: Check, if it's a library
        if not self.svm and isinstance(name, basestring):
            possibleNames = []
            if name in dir(sgClassifier):
                possibleNames = [name]
            possibleNames += [s for s in dir(sgClassifier) if s.lower()==name.lower()]
            
            if not len(possibleNames):
                msg = "Library '%s' is not known." % name
                raise mdp.NodeException(msg)

            # Take the first classifier which is callable
            # TODO: Evaluate error checking
            for s in possibleNames:
                self._classifier = getattr(sgClassifier, s)
                self.svm = self._classifier()
                break

        if self.svm is None:
            msg = "The classifier '%s' is not supported." %name
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
        'options' must be a tuple with the arguments of the kernel constructor in shogun.
        Therefore, in case of error, you will have to consult the shogun documentation.
        """
        if options is None:
            options = {}
        if name in ShogunSVMNode.kernel_parameters and not isinstance(options, list):
            default_opts = _OrderedDict(ShogunSVMNode.kernel_parameters[name])
            default_opts.update(options)
            options = default_opts._vals
        
        kernel_meth = getattr(sgKernel, name)
        self.kernel = kernel_meth(*options)
        
    def _stop_training(self):
        
        # self._normalize_labels()
        self._norm_labels = self._cl

        self.features = sgFeatures.RealFeatures(self._x.transpose())

        if issubclass(self._classifier, sgClassifier.LinearClassifier):
            self.svm.set_features(self.features)
        else:
            self.kernel.init(self.features, self.features)
            self.svm.set_kernel(self.kernel)
        
        # shogun expects floats
        labels = sgFeatures.Labels(self._norm_labels.astype(float))
        self.svm.set_labels(labels)
        
        #print "Training:"
        #print "Classifier: %s", self._classifier
        #print "Kernel: %s", self.kernel
        self.svm.train()

    def training_set(self, ordered=False):
        """Shows the set of data that has been inserted to be trained."""
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

    def _classify(self, x):
        """Classify the input data 'x'
        """

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
