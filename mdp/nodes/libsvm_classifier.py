import mdp
from mdp import numx

from svm_classifiers import _SVMClassifier, _LabelNormalizer

import svm as libsvm

class LibSVMClassifier(_SVMClassifier):
    """
    The LibSVMClassifier class acts as a wrapper around the LibSVM library
    for support vector machines.

    Information to the parameters can be found on
    http://www.csie.ntu.edu.tw/~cjlin/libsvm/
    """
    # The kernels and classifiers which LibSVM allows.
    kernels = ["RBF", "LINEAR", "POLY", "SIGMOID"]
    classifiers = ["C_SVC", "NU_SVC", "ONE_CLASS", "EPSILON_SVR", "NU_SVR"]

    def __init__(self, kernel=None, classifier=None, probability=True,
                 input_dim=None, output_dim=None, dtype=None):
        """
        probability -- Must be set to True, if algorithms based on probability
                       shall be used.
        """
        self.kernel_type = libsvm.RBF
        self._probability = probability
        self._classification_type = "multi"
        super(LibSVMClassifier, self).__init__(input_dim=input_dim,
                                               output_dim=output_dim,
                                               dtype=dtype)
        if kernel:
            self.set_kernel(kernel)
        if classifier:
            self.set_classifier(classifier)

    def _get_supported_dtypes(self):
        """Return the list of dtypes supported by this node."""
        # Support only float64 because of external library
        return ('float64',)

    def set_classifier(self, classifier):
        """
        Sets the classifier.

        classifier -- A string with the name of the classifier which
                      should be used. Possible values are in
                      self.classifiers
        """
        if classifier.upper() in self.classifiers:
            self.classifier_type = getattr(libsvm, classifier.upper())
        else:
            msg = "Classifier Type %s is unknown or not supported." % classifier
            raise TypeError(msg)

    def set_kernel(self, kernel):
        """
        Sets the kernel.

        kernel     -- A string with the name of the classifier which
                      should be used. Possible values are in
                      self.kernels
        """
        if kernel.upper() in self.kernels:
            self.kernel_type = getattr(libsvm, kernel.upper())
        else:
            msg = "Kernel Type %s is unknown or not supported." % kernel
            raise TypeError(msg)

    def _train_problem(self, labels, features, parameter):
        problem = libsvm.svm_problem(labels.tolist(), features.tolist())
        # Quieten libsvm
        # Method only available since libsvm 2.9 (on personal demand)
        try:
            libsvm.svmc.svm_set_quiet()
        except AttributeError:
            pass
        # Train
        model = libsvm.svm_model(problem, parameter)
        return model

    def _stop_training(self):
        super(LibSVMClassifier, self)._stop_training()
        self.normalizer = _LabelNormalizer(self.labels)

        if self._probability:
            prob = 1
        else:
            prob = 0
        self.parameter = libsvm.svm_parameter(svm_type=self.classifier_type,
                                              kernel_type=self.kernel_type,
                                              C=1, probability=prob)

        labels = self.normalizer.normalize(self.labels)
        features = self.data

        # Call svm training method.
        self.model = self._train_problem(labels, features, self.parameter)

    def _label(self, x):
        if isinstance(x, (list, tuple, numx.ndarray)):
            return numx.array([self.model.predict(xi) for xi in x])
        else:
            msg = "Data must be a sequence of vectors"
            raise mdp.NodeException(msg)

    def predict_probability(self, x):
        self._pre_execution_checks(x)
        if isinstance(x, (list, tuple, numx.ndarray)):
            return map(self.model.predict_probability, x)
        else:
            return self.model.predict_probability(x)

    def _prob(self, x):
        return [self.model.predict_probability(xi)[1] for xi in x]

    def _train(self, x, labels):
        super(LibSVMClassifier, self)._train(x, labels)
