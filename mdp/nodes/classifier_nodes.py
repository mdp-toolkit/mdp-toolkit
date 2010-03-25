import mdp
from mdp import numx as numx
import operator
import random
import itertools

class SignumClassifier(mdp.ClassifierNode):
    """This classifier node classifies as 1, if the sum of the data points is positive
        and as -1, if the data point is negative"""
    def is_trainable(self):
        return False
    
    def _classify(self, x):
        ret = [xi.sum() for xi in x]
        return numx.sign(ret)
    

class PerceptronClassifier(mdp.ClassifierNode):
    """A simple perceptron with input_dim input nodes."""
    def __init__(self, input_dim=None, dtype=None):
        super(PerceptronClassifier, self).__init__(input_dim, None, dtype)
        self.weights = []
        self.offset_weight = 0
        self.learning_rate = 0.1
    
    def _check_train_args(self, x, cl):
        if (isinstance(cl, (list, tuple, numx.ndarray)) and
            len(cl) != x.shape[0]):
            msg = ("The number of labels should be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)
        
        if (not isinstance(cl, (list, tuple, numx.ndarray))):
            cl = [cl]

        if (not numx.all(map(lambda x: abs(x) == 1, cl))):
            msg = "The labels must be either -1 or 1."
            raise mdp.TrainingException(msg)

    def _train(self, x, cl):
        """Update the internal structures according to the input data 'x'.
        
        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """

        # if weights are not yet initialised, initialise them
        if not len(self.weights):
            self.weights = numx.ones(self.input_dim)
        
        for xi, cli in mdp.utils.izip_stretched(x, cl):
            new_weights = self.weights
            new_offset = self.offset_weight

            rate = self.learning_rate * (cli - self._classify(xi))
            for j in range(self.input_dim):
                new_weights[j] = self.weights[j] + rate * xi[j]

            # the offset corresponds to a node with input 1 all the time
            new_offset = self.offset_weight + rate * 1

            self.weights = new_weights
            self.offset_weight = new_offset

    def _classify(self, x):
        """Classifies the perceptron.
        """
        return numx.sign(numx.dot(x, self.weights) + self.offset_weight)


class NaiveBayesClassifier(mdp.ClassifierNode):
    """A naive Bayes Classificator.
    In order to be usable for spam filtering, the words must be transformed
    with a hash function.
    
    Right now, it is only a two-class model. If needed, it can be possible to
    allow for multiple class training and classification.
    """
    def __init__(self, input_dim=None, dtype=None):
        super(NaiveBayesClassifier, self).__init__(input_dim, None, dtype)
        self.nospam = {}
        self.spam = {}
        self.num_nospam = 0
        self.num_spam = 0
    
    def _check_train_args(self, x, cl):
        if (isinstance(cl, (list, tuple, numx.ndarray)) and
            len(cl) != x.shape[0]):
            msg = ("The number of labels should be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)
        
        if (not isinstance(cl, (list, tuple, numx.ndarray))):
            cl = [cl]

        if (not all(map(lambda x: abs(x) == 1, cl))):
            msg = "The labels must be either -1 or 1."
            raise mdp.TrainingException(msg)

    def _train(self, x, cl):
        """Update the internal structures according to the input data 'x'.
        
        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """
        for xi, cli in mdp.utils.izip_stretched(x, cl):
            self._learn(xi, cli)
            
        # clear input dim hack
        self._set_input_dim(None)

    def _learn(self, words, cl):
        # assume words is iterable of ints, cl is number
        if cl == -1:
            # mail is spam
            self.num_spam += 1
            for word in words:
                if word in self.spam:
                    self.spam[word] += 1
                else:
                    self.spam[word] = 1
        elif cl == 1:
            # mail is not spam
            self.num_nospam += 1
            for word in words:
                if word in self.nospam:
                    self.nospam[word] += 1
                else:
                    self.nospam[word] = 1
                    
    def _prob(self, words):
        # TODO: What happens when values are twice in prob_sort
        
        # clear input dim hack
        self._set_input_dim(None)
        
        prob_spam = [] # P(W|Spam)
        prob_nospam = []
        for word in words[0,:]: # FIXME
            if word in self.spam:
                prob_spam.append(float(self.spam[word]) / self.num_spam)
            else:
                # take a minimum value to avoid multiplication with 0 on unknown words
                prob_spam.append(0.5 / self.num_spam)
                
            if word in self.nospam:
                prob_nospam.append(float(self.nospam[word]) / self.num_nospam)
            else:
                prob_nospam.append(0.5 / self.num_nospam)

        num_classifiers_spam = 5 # take the n best values of spam
        num_classifiers_nospam = 5 # and of nospam
        
        prob = zip(prob_nospam, prob_spam)
        
        prob.sort(key = operator.itemgetter(1), reverse=True) # investigate the pre-sorting
        prob.sort(key = operator.itemgetter(0), reverse=True)

        all_prob = prob[0:num_classifiers_nospam]
        
        prob.sort(key = operator.itemgetter(0), reverse=True)
        prob.sort(key = operator.itemgetter(1), reverse=True)
        all_prob += prob[0:num_classifiers_spam]
        
        p_spam = float(self.num_spam) / (self.num_nospam + self.num_spam)
        p_nospam = float(self.num_nospam) / (self.num_nospam + self.num_spam)
        
        p_spam_W = p_spam
        p_nospam_W = p_nospam

        for t in all_prob:
            p_nospam_W *= t[0]
            p_spam_W *= t[1]
            
        # all_prob are not independent, so normalise it
        p_sum = p_spam_W + p_nospam_W
        p_spam_W = p_spam_W / p_sum
        p_nospam_W = p_nospam_W / p_sum
        
        return { -1: p_spam_W, 1: p_nospam_W }
                    
    def _classify(self, words):
        """Classifies the words.
        """
        # clear input dim hack
        self._set_input_dim(None)
        
        p = self._prob(words)
        p_spam_W = p[-1]
        p_nospam_W = p[1]
        try:
            q = p_spam_W / p_nospam_W
        except ZeroDivisionError:
            return 1
        print "q =", q
        return - numx.sign(q - 1)


class SimpleMarkovClassifier(mdp.ClassifierNode):
    """A simple version of a Markov classifier.
    It can be trained on a vector of tuples the label being the next element
    in the testing data.
    """
    def __init__(self, input_dim=None, dtype=None):
        super(SimpleMarkovClassifier, self).__init__(input_dim, None, dtype)
        self.ntotal_connections = 0

        self.features = {}
        self.labels = {}
        self.connections = {}

    def _check_train_args(self, x, cl):
        if (isinstance(cl, (list, tuple, numx.ndarray)) and
            len(cl) != x.shape[0]):
            msg = ("The number of labels should be equal to the number of "
                   "datapoints (%d != %d)" % (len(cl), x.shape[0]))
            raise mdp.TrainingException(msg)
        
        if (not isinstance(cl, (list, tuple, numx.ndarray))):
            cl = [cl]

    def _train(self, x, cl):
        """Update the internal structures according to the input data 'x'.
        
        x -- a matrix having different variables on different columns
             and observations on the rows.
        cl -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """
        # if cl is a number, all x's belong to the same class
        for xi, cli in mdp.utils.izip_stretched(x, cl):
            self._learn(xi, cli)
    
    def _learn(self, feature, label):
        feature = tuple(feature)
        self.ntotal_connections += 1

        if label in self.labels:
            self.labels[label] += 1
        else:
            self.labels[label] = 1
        
        if feature in self.features:
            self.features[feature] += 1
        else:
            self.features[feature] = 1

        connection = (feature, label)
        if connection in self.connections:
            self.connections[connection] += 1
        else:
            self.connections[connection] = 1

    def _prob(self, features):
        return [self._prob_one(feature) for feature in features]
    
    def _prob_one(self, feature):
        feature = tuple(feature)
        probabilities = {}
        
        try:
            n_feature_connections = self.features[feature]
        except KeyError:
            n_feature_connections = 0
            # if n_feature_connections == 0, we get a division by zero
            # we could throw here, but maybe it's best to simply return
            # an empty dict object
            return {}
        
        for label in self.labels:
            conn = (feature, label)
            try:
                n_conn = self.connections[conn]
            except KeyError:
                n_conn = 0
            
            try:
                n_label_connections = self.labels[label]
            except KeyError:
                n_label_connections = 0
            
            p_feature_given_label = 1.0 * n_conn / n_label_connections
            p_label = 1.0 * n_label_connections / self.ntotal_connections
            p_feature = 1.0 * n_feature_connections / self.ntotal_connections
            prob = 1.0 * p_feature_given_label * p_label / p_feature
            probabilities[label] = prob
        return probabilities
    

class DiscreteHopfieldClassifier(mdp.ClassifierNode):
    """Node for simulating a simple discrete Hopfield model"""
    # TODO: It is unclear if this belongs to classifiers or is a general node
    # because label space is a subset of feature space
    def __init__(self, input_dim=None):
        dtype = bool
        super(DiscreteHopfieldClassifier, self).__init__(input_dim, input_dim, dtype)
        self._weight_matrix = 0 # assigning zero to ease addition
        self._num_patterns = 0
        self._shuffled_update = True
    
    def _train(self, x):
        """Provide the hopfield net with the possible states.
        
        x -- a matrix having different variables on different columns
            and observations on rows.
        """
        for pattern in x:
            self._train_one(pattern)
    
    def _train_one(self, pattern):
        pattern = mdp.utils.bool_to_sign(pattern)
        weights = numx.outer(pattern, pattern)
        self._weight_matrix += weights / float(self.input_dim)
        self._num_patterns += 1
    
    @property
    def memory_size(self):
        """Returns the Hopfield net's memory size"""
        return self.input_dim
    
    @property
    def load_parameter(self):
        """Returns the load parameter of the Hopfield net.
        The quality of memory recall for a Hopfield net breaks down when the
        load parameter is larger than 0.14."""
        return self._num_patterns / float(self.input_dim)
    
    def _stop_training(self):
        # remove self-feedback
        # we could use numx.fill_diagonal, but thats numpy 1.4 only
        for i in range(self.input_dim):
            self._weight_matrix[i][i] = 0
    
    def _classify(self, x, threshold = 0):
        """Retrieves patterns from the associative memory.
        """
        threshold = numx.zeros(self.input_dim) + threshold
        return numx.array([self._classify_one(pattern, threshold) for pattern in x])
    
    def _classify_one(self, pattern, threshold):
        pattern = mdp.utils.bool_to_sign(pattern)    
        
        has_converged = False
        while not has_converged:
            has_converged = True
            iter_order = range(len(self._weight_matrix))
            if self._shuffled_update:
                numx.random.shuffle(iter_order)
            for row in iter_order:
                w_row = self._weight_matrix[row]
                
                thresh_row = threshold[row]
                new_pattern_row = numx.sign(numx.dot(w_row, pattern) - thresh_row)
                
                if new_pattern_row == 0:
                    # Following McKay, Neural Networks, we do nothing
                    # when the new pattern is zero
                    pass
                elif pattern[row] != new_pattern_row:
                    has_converged = False
                    pattern[row] = new_pattern_row
        return mdp.utils.sign_to_bool(pattern)


class KMeansClassifier(mdp.ClassifierNode):
    def __init__(self, num_clusters, input_dim=None, dtype=None):
        """Simple classifier for K-Means clustering.
        
        num_clusters -- number of centroids to use = number of clusters
        """
        super(KMeansClassifier, self).__init__(input_dim, None, dtype)
        self._num_clusters = num_clusters
        self._data = None
        self._centroids = None
    
    def _train(self, x):
        # append all data
        if self._data is None:
            self._data = x
        else:
            self._data = numx.concatenate((self._data, x))
    
    def _stop_training(self):
        num_points = len(self._data)
        # choose initial centroids unless they are already given
        if not self._centroids:
            centr_idx = random.sample(xrange(len(self._data)), self._num_clusters)
            centroids = self._data[centr_idx]
        else:
            centroids = self._centroids
        
        while True:
            # list of (sum_position, num_clusters)
            new_centroids = [(0, 0)] * len(centroids)
            # cluster
            for x in self._data:
                idx = self._nearest_centroid_idx(x, centroids)
                # update position and count
                pos_count = (new_centroids[idx][0] + x, new_centroids[idx][1] + 1)
                new_centroids[idx] = pos_count
            
            # get new centroid position
            new_centroids = numx.array([c[0] / c[1] for c in new_centroids])
            # check if we are stable
            if numx.all(new_centroids == centroids):
                self._centroids = centroids
                return
            centroids = new_centroids
            
    def _nearest_centroid_idx(self, data, centroids):
        dists = [numx.linalg.norm(data - c) for c in centroids]
        # return index
        return min(zip(dists, itertools.count()))[1] 

    def _classify(self, x):
        """For a set of feature vectors x, this classifier returns
        a list of centroids.
        """
        return [self._nearest_centroid_idx(xi, self._centroids) for xi in x]
