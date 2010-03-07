import mdp
from mdp import numx

from svm_classifiers import _SVMClassifier, _LabelNormalizer

import svm as libsvm

class LibSVMClassifier(_SVMClassifier):

    kernels = ["RBF", "LINEAR", "POLY", "SIGMOID"]
    classifiers = ["C_SVC", "NU_SVC", "ONE_CLASS", "EPSILON_SVR", "NU_SVR"]

    def __init__(self, probability=True, input_dim=None, dtype=None):
        """
        probability -- Must be set to True, if algorithms based on probability shall be used.
        """        
        self.kernel_type = libsvm.RBF
        self._probability = probability
        self._classification_type = "multi"
        super(LibSVMClassifier, self).__init__(input_dim=input_dim, dtype=dtype)

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
        self.normalizer = _LabelNormalizer(self._in_labels)
        
        if self._probability:
            prob = 1
        else:
            prob = 0
        self.parameter = libsvm.svm_parameter(svm_type=self.classifier_type,
                                              kernel_type=self.kernel_type,
                                              C=1, probability=prob)

        labels = self.normalizer.normalize(self._in_labels)
        features = self._in_features

        # Call svm training method.
        self.model = self._train_problem(labels, features, self.parameter)

    def _classify(self, x):
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

    def _train(self, x, cl):
        super(LibSVMClassifier, self)._train(x, cl)

