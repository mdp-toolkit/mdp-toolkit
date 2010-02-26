import mdp
from mdp import numx, numx_rand

from svm_nodes import _SVMNode

import svm as libsvm

class LibSVMNode(_SVMNode):

    kernels = ["RBF", "LINEAR", "POLY", "SIGMOID"]
    classifiers = ["C_SVC", "NU_SVC", "ONE_CLASS", "EPSILON_SVR", "NU_SVR"]

    def __init__(self, probability=True, input_dim=None, dtype=None):
        """
        probability -- Shall the probability be computed
        """
        self.normalize = False
        
        self.kernel_type = libsvm.RBF
        self._probability = probability
        self._classification_type = "multi"
        super(LibSVMNode, self).__init__(input_dim=input_dim, dtype=dtype)

    def set_classifier(self, classifier):
        if classifier.upper() in self.classifiers:
            self.classifier_type = getattr(libsvm, classifier.upper())
        else:
            msg = "Classifier Type %s is unknown or not supported." % classifier
            raise TypeError(msg)

    def set_kernel(self, kernel):
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
        if self.normalize:
            self._normalize_labels()
        
        if self._probability:
            prob = 1
        else:
            prob = 0
        self.parameter = libsvm.svm_parameter(kernel_type = self.kernel_type, C=1, probability=prob)

        labels = self._cl
        features = self._x

        # Call svm training method.
        self.model = self._train_problem(labels, features, self.parameter)

    def classify(self, x):
        self._pre_execution_checks(x)
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

    def _train(self, x, cl):
        super(LibSVMNode, self)._train(x, cl)
