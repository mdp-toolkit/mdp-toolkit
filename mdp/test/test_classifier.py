# -*- coding: utf-8 -*-

"""These are test functions for MDP classifiers.

Run them with:
>>> import mdp
>>> mdp.test("classifier")

"""
import unittest
import inspect

import mdp
from mdp import numx, numx_rand
from mdp import ClassifierNode
from mdp.nodes import (SignumClassifier, PerceptronClassifier,
                       SimpleMarkovClassifier, DiscreteHopfieldClassifier,
                       KMeansClassifier)
from mdp.utils import weighted_choice

def _sigmoid(t):
    return 1.0 / (1.0 + numx.exp(-t))

class _BogusClassifier(ClassifierNode):
    def is_trainable(self):
        return False
    def _label(self, x):
        return [r[0] for r in self.rank(x)]
    def _prob(self, x):
        return [{-1: _sigmoid(sum(xi)), \
                  1: 1 - _sigmoid(sum(xi))} for xi in x]


class ClassifierTestSuite(unittest.TestSuite):
    def __init__(self, testname=None):
        unittest.TestSuite.__init__(self)

        if testname is not None:
            self._utils_test_factory([testname])
        else:
            # get all tests
            self._utils_test_factory()

    def _utils_test_factory(self, methods_list=None):
        if methods_list is None:
            methods_list = dir(self)
        for methname in methods_list:
            try:
                meth = getattr(self,methname)
            except AttributeError:
                continue
            if inspect.ismethod(meth) and meth.__name__[:4] == "test":
                # create a nice description
                descr = 'Test '+(meth.__name__[4:]).replace('_',' ')
                self.addTest(unittest.FunctionTestCase(meth,
                             description=descr))
    
    def testClassifierNode_ranking(self):
        bc = _BogusClassifier()
        test_data = numx_rand.random((30, 20)) - 0.5
        for r, p in zip(bc.rank(test_data), bc.prob(test_data)):
            # check that the ranking order is correct
            assert p[r[0]] >= p[r[1]], "Rank returns labels in incorrect order"
            # check that the probabilities sum up to 100
            assert 0.999 < p[r[0]] + p[r[1]] < 1.001
        
    
    def testSignumClassifier(self):
        c = SignumClassifier()
        res = c.label(mdp.numx.array([[1, 2, -3, -4], [1, 2, 3, 4]]))
        assert c.input_dim == 4
        assert res.tolist() == [-1, 1]
        
    def testPerceptronClassifier(self):
        or_Classifier = PerceptronClassifier()
        for i in range(100):
            or_Classifier.train(mdp.numx.array([[0, 0]]), -1)
            or_Classifier.train(mdp.numx.array([[0, 1], [1, 0], [1, 1]]), 1)
        assert or_Classifier.input_dim == 2
        
        res = or_Classifier.label(mdp.numx.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        assert res.tolist() == [-1, 1, 1, 1]
                            
        and_Classifier = PerceptronClassifier()
        for i in range(100):
            and_Classifier.train(mdp.numx.array([[0, 0], [0, 1], [1, 0]]), -1)
            and_Classifier.train(mdp.numx.array([[1, 1]]), 1)
        res = and_Classifier.label(mdp.numx.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        assert res.tolist() == [-1, -1, -1, 1]
                            
        xor_Classifier = PerceptronClassifier()
        for i in range(100):
            xor_Classifier.train(mdp.numx.array([[0, 0], [1, 1]]), -1)
            xor_Classifier.train(mdp.numx.array([[0, 1], [1, 0]]), 1)
        res = xor_Classifier.label(mdp.numx.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        assert res.tolist() != [-1, 1, 1, -1], "Something must be wrong here. XOR is impossible in a single-layered perceptron."
        

    def testSimpleMarkovClassifier(self):
        mc = SimpleMarkovClassifier(dtype="unicode")
        text = "after the letter e follows either space or the letters r t or i"
        
        for word in text.split():
            word = word.lower()

            features = zip(" " + word)
            labels = list(word + " ")

            mc.train(mdp.numx.array(features), labels)
        
        assert mc.input_dim == 1
        
        num_transitions = 0
        features = mc.features
        for feature, count in features.items():
            if count:
                prob = mc.prob(mdp.numx.array([feature]))
                prob_sum = 0
                for p in prob:
                    for k, v in p.items():
                        prob_sum += v
                        if v:
                            num_transitions += 1
                            #print "".join(feature).replace(" ", "_"), "->", k, "(", v, ")"
                assert abs(prob_sum - 1.0) < 1e-5
        
        # calculate the number of transitions (the negative set deletes the artefact of two spaces)
        trans = len(set((zip("  ".join(text.split()) + " ", " " + "  ".join(text.split())))) - set([(' ', ' ')]))
        assert num_transitions == trans
        
        letters_following_e = [' ', 'r', 't', 'i']
        letters_prob = mc.prob(mdp.numx.array([['e']]))[0]
        prob_sum = 0
        for letter, prob in letters_prob.items():
            prob_sum += prob
            if prob > 1e-5:
                assert letter in letters_following_e
        
        assert abs(prob_sum - 1.0) < 1e-5
    
    def testDiscreteHopfieldClassifier(self):
        h = DiscreteHopfieldClassifier()
        
        memory_size = 100
        patterns = numx.array(
                   [numx.sin(numx.linspace(0, 100 * numx.pi, memory_size)) > 0,
                    numx.sin(numx.linspace(0, 50 * numx.pi, memory_size)) > 0,
                    numx.sin(numx.linspace(0, 20 * numx.pi, memory_size)) > 0,
                    numx.sin(numx.linspace(0, 15 * numx.pi, memory_size)) > 0,
                    numx.sin(numx.linspace(0, 10 * numx.pi, memory_size)) > 0,
                    numx.sin(numx.linspace(0, 5 * numx.pi, memory_size)) > 0,
                    numx.sin(numx.linspace(0, 2 * numx.pi, memory_size)) > 0
                    ])
        h.train(patterns)
        h.input_dim = memory_size
        
        for p in patterns:
            # check if patterns are fixpoints
            assert numx.all(p == h.label(numx.array([p])))
        
        for p in patterns:
            # check, if a noisy pattern is recreated
            noisy = numx.array(p)
            for i in range(len(noisy)):
                if numx.random.random() > 0.9:
                    noisy[i] = not noisy[i]
            retrieved = h.label(numx.array([noisy]))
            # Hopfield nets are blind for inversion, need to check either case
            assert numx.all(retrieved == p) or numx.all(retrieved != p)
    
    def testKMeansClassifier(self):
        num_centroids = 3
        k = KMeansClassifier(num_centroids)
        a = numx.random.rand(50, 2)
        k.train(a)
        res = k.label(a)
        
        # check that the number of centroids is correct
        assert len(set(res)) == num_centroids
        
        k = KMeansClassifier(2)
        a1 = numx.random.rand(50, 2) - 1
        a2 = numx.random.rand(50, 2) + 1
        k.train(a1)
        k.train(a2)
        res1 = k.label(a1)
        res2 = k.label(a2)
        # check that both clusters are completely identified and different
        assert len(set(res1)) == 1 and len(set(res2)) == 1 and set(res1) != set(res2), \
            "Error in K-Means classifier. This might be a bug or just a local minimum."
        

def get_suite(testname=None):
    return ClassifierTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

