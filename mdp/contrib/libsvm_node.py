import mdp
from mdp import numx, numx_rand

from svm_nodes import _SVMNode

try:
    import svm as libsvm
except ImportError:
    pass

class LibSVMNode(_SVMNode):
    """
        Problems with LibSVM:
        - Screen output can only be disabled when a #if 1 clause in the cpp file is disabled
    
    """

    kernels = ["RBF", "LINEAR", "POLY", "SIGMOID"]

    def __init__(self, probability=True, input_dim=None, dtype=None):
        """
        probability -- Shall the probability be computed
        """
        try:
            libsvm
        except NameError:
            msg = "Using LibSVMNode requires the python binding of LibSVM to be installed."
            raise ImportError(msg)

        self.kernel_type = libsvm.RBF
        self._probability = probability
        self._classification_type = "multi"
        super(LibSVMNode, self).__init__(input_dim=input_dim, dtype=dtype)

    def setKernel(self, kernel):
        if kernel.upper() in self.kernels:
            self.kernel_type = getattr(libsvm, kernel.upper())
        else:
            msg = "Kernel Type %s is unknown." % kernel
            raise TypeError(msg)

    def _cross_validation(self, num_parts, parameter):
        labels = self._norm_labels
        features = self._x

        # need to shuffle indices to avoid degeneration because of ordered input
        shuffled = numx.arange(labels.size)
        numx_rand.shuffle(shuffled)
        shuffled_parts = numx.array_split(shuffled, num_parts)
        testerr = 0.0
        for i in range(0, len(shuffled_parts)):
            # build up indices
            i_test = shuffled_parts[i]
            i_train = numx.array([], dtype='int')
            for j in range(0, len(shuffled_parts)):
                if j == i: continue
                i_train = numx.concatenate( (i_train, shuffled_parts[j]) )

            model = self._train_problem(labels[i_train], features[i_train], parameter)
            test_labels = labels[i_test] # .tolist()
            test_features = features[i_test]

            result = map(model.predict, test_features)
            testerr +=  sum((result!=test_labels)*1)

        print "Testing Error is", testerr / labels.size
        #self._train_problem(labels, features, parameter)

    def _train_problem(self, labels, features, parameter):
        problem = libsvm.svm_problem(labels.tolist(), features.tolist())
        model = libsvm.svm_model(problem, parameter)
        return model

    def _stop_training(self):
        self._normalize_labels()

        if self._probability:
            prob = 1
        else:
            prob = 0
        self.parameter = libsvm.svm_parameter(kernel_type = self.kernel_type, C=1, probability=prob)

        labels = self._norm_labels
        features = self._x
        # Call svm training method.
        self.model = self._train_problem(labels, features, self.parameter)

    def classify(self, x):
        if isinstance(x, (list, tuple, numx.ndarray)):
            return map(self.model.predict, x)
        else:
            return self.model.predict(x)

    def probability(self, x):
        print self.model
        self._pre_execution_checks(x)
        if isinstance(x, (list, tuple, numx.ndarray)):
            return map(self.model.predict_probability, x)
        else:
            return self.model.predict_probability(x)

    def _train(self, x, cl):
        super(LibSVMNode, self)._train(x, cl)

    def grid_parameter_search(self, param_range):
        # TODO: Cross-validation
        params = []
        sizes = []
        for p in param_range.keys():
            params.append(p)
            sizes.append(len(param_range[p]))
        grid_error = numx.zeros(tuple(sizes)) * numx.nan

        for position, value in numx.ndenumerate(grid_error):
            for i in len(position):
                param = params[i]
                param_val = param_range[param][position[i]]
                # set_param(param, param_val)
            #execute
            #grid_error[position] = error_res
