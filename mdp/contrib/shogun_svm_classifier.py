import mdp
from mdp import numx

from svm_classifiers import _SVMClassifier, _LabelNormalizer

import shogun.Kernel as sgKernel
import shogun.Features as sgFeatures
import shogun.Classifier as sgClassifier

# We need to have at least SHOGUN 0.9, as we rely on
# SHOGUN's CClassifier::classify() method.
# (It makes our code much nicer, by the way.)
#
try:
    version = sgKernel._Kernel.Version_get_version_release()
except AttributeError:
    version = ""

if not (version.startswith('v0.9') or version.startswith('v1.')):
    msg = "We need at least SHOGUN version 0.9."
    raise ImportError(msg)

# switch off spurious warnings from shogun
import warnings
warnings.filterwarnings('ignore',
                        '.*Perceptron algorithm did not converge after.*',
                        RuntimeWarning)

# maybe integrate to the class
def is_shogun_classifier(test_classifier):
    """Check, if a class is a subclass of a SHOGUN classifier."""
    try:
        return issubclass(test_classifier, sgClassifier.Classifier)
    except (TypeError, NameError):
        # need to fetch NameError for some swig reasons
        return False
    
default_shogun_classifiers = []
for labels in dir(sgClassifier):
    test_classifier = getattr(sgClassifier, labels)
    if is_shogun_classifier(test_classifier):
        default_shogun_classifiers.append(test_classifier)

shogun_classifier_types = {}
for ct in dir(sgClassifier):
    if ct.startswith("CT_"):
        shogun_classifier_types[getattr(sgClassifier, ct)] = ct


class _OrderedDict(object):
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
                if len(entry) > 1: 
                    new_val = entry[1]
                else:
                    None
            if new_key in self._keys:
                i = self._keys.index(new_key)
                self._vals[i] = new_val
            else:
                self._keys.append(new_key)
                self._vals.append(new_val)
    
    @property
    def values(self):
        return self._vals


class Classifier(object):
    """This Classifier class hides the logic of setting a classifier and kernel
    from the main classifier node.
    It is currently not intended to be used on its own and might, in fact, get
    completely integrated into the ShogunSVMClassifier class."""
    
    def __init__(self):
        self._class = None
        self._instance = None
        
    def set_classifier(self, classifier, args=None):
        """Sets and initialises the classifier. If a classifier is reset
        by the user, the parameters will have to be set again.
        'classifier' can be a string, a subclass of shogun.Classifier
        or an instance of such a class.
        Note that some classifiers require initialisation arguments
        while others don't. Also note that neither labels nor feature
        vectors should be given with 'args'. 
        
        classifier -- the SHOGUN classifier to use
        args       -- the list of arguments needed for the SHOGUN
                      constructor method of the classifier
        """
        if args is None:
            args = []
        self._class = None
        self._instance = None
        
        # If name is a classifier instance: Take it
        if isinstance(classifier, sgClassifier.Classifier):
            self._class = classifier.__class__
            self._instance = classifier
        
        # If name is a class: Make an instance
        elif classifier.__class__ == type:
            if is_shogun_classifier(classifier):
                try:
                    self._class = classifier
                    self._instance = self._class(*args)
                except AttributeError:
                    msg = "Library '%s' could not be instantiated. Abstract class?" % classifier
                    raise mdp.NodeException(msg)
            else:
                msg = "The classifier '%s' is no subclass of CClassifier." % self._class.__name__
                raise mdp.NodeException(msg)

        # If classifier is a string: Check, if it's the name of a default library
        elif isinstance(classifier, basestring):
            possibleClasses = [labels for labels in default_shogun_classifiers
                               if labels.__name__.lower() == classifier.lower()]

            if not len(possibleClasses):
                msg = "Library '%s' is not a known subclass of CClassifier." % classifier
                raise mdp.NodeException(msg)

            # Take the first classifier which works
            for labels in possibleClasses:
                try:
                    self._instance = labels(*args)
                    self._class = labels
                except AttributeError:
                    # we might have a virtual class here
                    pass
            if not self._instance:
                msg = "Library '%s' could not be instantiated. Abstract class?" % classifier
                raise mdp.NodeException(msg)

        if not self._class or not self._instance:
            msg = "The classifier '%s' is not supported." % classifier
            raise mdp.NodeException(msg)
        
        if self.classifier_type() == "CT_NONE":
            msg = "The classifier '%s' is not valid." % classifier
            raise mdp.NodeException(msg) 
            
    
    def classifier_type(self):
        """Returns the SHOGUN classifier type as a string."""
        return shogun_classifier_types[self._instance.get_classifier_type()]
    
    def set_param(self, param, *value):
        """Sets parameters for the classifier.
        
        This calls set_param(*value) on the classifier instance.
        """
        # Non-standard cases
        if param == "C" and len(value) == 1:
            # set_C needs two arguments, but if we get only one,
            # we call set_C(arg, arg)
            value += value
        # get the parameter setting method
        meth = getattr(self._instance, "set_" + param)
        # call it 
        meth(*value)
        
    def get_param(self, param, *args):
        """Returns the parameter for a with a given name.
        
        This calls get_param(*args) on the classifier instance.
        
        """
        meth = getattr(self._instance, "get_" + param)
        return meth(*args)

    def set_train_features(self, features, labels):
        if issubclass(self._class, sgClassifier.LinearClassifier):
            self._instance.set_features(features)
        elif issubclass(self._class, sgClassifier.CKernelMachine):
            try:
                self.kernel.init(features, features)
            except AttributeError:
                msg = "You must assign a kernel before training."
                raise mdp.NodeException(msg)
            #self.svm.set_kernel(self.kernel)
        else:
            msg = "Sorry, SHOGUN classifiers of this type are not yet implemented"
            raise mdp.NodeException(msg)
        
        self._instance.set_labels(labels)
    
    def train(self):
        self._instance.train()
    
    def label(self, test_features):
        return self._instance.classify(test_features).get_labels()
    
    @property
    def takes_kernel(self):
        """Returns true, if the current classifier is a kernel machine."""
        return issubclass(self._class, sgClassifier.CKernelMachine)
    
    def _get_kernel(self):
        """Retrieve the currently set kernel from the classifier instance."""
        try:
            return self._instance.get_kernel()
        except AttributeError:
            msg = "Error retrieving kernel. The classifier might not support kernels."
            raise mdp.NodeException(msg)
    
    def _set_kernel(self, kernel):
        """Set the kernel in the current classifier instance"""
        try:
            self._instance.set_kernel(kernel)
        except AttributeError:
            msg = "Error setting kernel. The classifier might not support kernels."
            raise mdp.NodeException(msg)
    
    kernel = property(_get_kernel, _set_kernel)
    
    def __str__(self):
        str = self._instance.__class__.__name__
        try:
            str = str + " with %s kernel" % self.kernel.get_name()
        except (mdp.NodeException, AttributeError):
            pass
        return str


class ShogunSVMClassifier(_SVMClassifier):
    """The ShogunSVMClassifier works as a wrapper class for accessing
    the SHOGUN machine learning toolbox for support vector machines.
    
    Most kernel machines and linear classifier should work with this class.
    
    Currently, distance machines such as the K-means classifier
    are not supported yet.
    
    Information to paramters and additional options can be found on
    http://www.shogun-toolbox.org/
    
    Note that some parts in this classifier might receive some
    refinement in the future.
    """

    default_parameters = {
        'C': 1,
        'epsilon': 1e-3,
    }

    # Swig-code does not work with named parameters, so we have to define an order
    kernel_parameters = {
        # Simple float64t kernels
        'Chi2Kernel': [('size', 10), ('width', 1.4)],
        'GaussianKernel': [('size', 10), ('width', 1.9)],
        'LinearKernel': [],
        'PolyKernel': [('size', 10), ('degree', 3), ('inhomogene', True)],
        'PyramidChi2': [('size',), ('num_cells2',),
                        ('weights_foreach_cell2',), ('width_computation_type2',),
                        ('width2',)],
        'SigmoidKernel': [('size', 10), ('gamma', 1), ('coef0', 0)]
    }

    def __init__(self, classifier="libsvmmulticlass", classifier_arguments=(),
                 classifier_options=None,
                 num_threads="autodetect", input_dim=None, dtype=None):
        """
        Initialises a new ShogunSVMClassifier.
        
        Keyword arguments:
            
            classifier  -- The classifier to use
            classifier_arguments -- Arguments needed for the constructor of the classifier
            classifier_options -- Options for the classifier
            num_threads -- The number of threads, SHOGUN should use
                           can be set to "autodetect", then SHOGUN will use
                           the number of CPU cores.
                           Attention: this could crash on windows
        
        """
        super(ShogunSVMClassifier, self).__init__(input_dim=input_dim, dtype=dtype)
        
        if classifier_options is None:
            classifier_options = {}

        self.classifier = Classifier()
        self.classifier.set_classifier(classifier, classifier_arguments)
        self.classifier_options = self.default_parameters
        self.classifier_options.update(classifier_options)

        for p in self.classifier_options.keys():
            try:
                self.set_classifier_param(p, self.classifier_options[p])
            except:
                pass
            
        self._num_threads = num_threads
        self._set_num_threads()

    def _set_num_threads(self):
        # init number of threads
        if self._num_threads == "autodetect":
            try:
                self._num_threads = self.classifier._instance.parallel.get_num_cpus()
            except SystemError:
                # We're helping shogun here
                self._num_threads = 1
        self.classifier._instance.parallel.set_num_threads(self._num_threads)

    def set_classifier_param(self, param, *value):
        """Sets parameters for the classifier.
        """
        self.classifier.set_param(param, *value)

    def set_kernel(self, kernel_name, kernel_options=None):
        """Sets the Kernel along with options.
        'options' must be a tuple with the arguments of the kernel constructor
        in SHOGUN.
        We try to guess it right in many cases but in general, you will have to
        consult the SHOGUN documentation.
        
        kernel    --    The kernel to use. Default parameters are specified for
                            "PolyKernel"
                            "GaussianKernel"
                            "LinearKernel"
                            "SigmoidKernel"
                        Further kernels are possible if they are included in
                        SHOGUN and if kernel_options provides the correct init
                        arguments.
        kernel_options -- For known kernels, a dict specifying the options is
                          possible. Options not included take a default value.
                          Unknown kernels need an ordered list of constructor
                          arguments.
        """
        if kernel_options is None:
            kernel_options = {}
        if kernel_name in ShogunSVMClassifier.kernel_parameters \
            and not isinstance(kernel_options, list):
            default_opts = _OrderedDict(ShogunSVMClassifier.kernel_parameters[kernel_name])
            default_opts.update(kernel_options)
            options = default_opts.values
        
        kernel_meth = getattr(sgKernel, kernel_name)
        try:
            kernel = kernel_meth(*options)
        except NotImplementedError, msg:
            msg = ("Tried to call %s with arguments %s\n" %
                   (kernel_meth.__module__ + '.' + kernel_meth.__name__,
                    tuple(options).__repr__()) +
                   "Got the following error message:\n" + msg.__str__())
            raise mdp.NodeException(msg) 
        self.classifier.kernel = kernel
    
    def _stop_training(self):
        super(ShogunSVMClassifier, self)._stop_training()
        self.normalizer = _LabelNormalizer(self.labels)
        labels = self.normalizer.normalize(self.labels)
        # shogun expects float labels
        labels = sgFeatures.Labels(labels.astype(float))
        
        features = sgFeatures.RealFeatures(self.data.transpose())
        
        self.classifier.set_train_features(features, labels)
        self.classifier.train()
    
    def training_set(self, ordered=False):
        """Shows the set of data that has been inserted to be trained."""
        if ordered:
            labels = set(self.labels)
            data = {}
            for l in labels:
                data[l] = []
            for k, v in zip(self.labels, self.data):
                data[k].append(v)
            return data
        else:
            return zip(self.labels, self.data)
    
    def _label(self, x):
        """Classify the input data 'x'
        """
        test_features = sgFeatures.RealFeatures(x.transpose())

        labels = self.classifier.label(test_features)
        
        if self.normalizer:
            return self.normalizer.revert(labels)
        else:
            return labels
