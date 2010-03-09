# -*- coding: utf-8 -*-

"""These are test functions for MDP classifiers.

Run them with:
>>> import mdp
>>> mdp.test("classifier")

"""
import unittest
import inspect

import mdp
from mdp import numx
from mdp.nodes import (SignumClassifier, PerceptronClassifier, NaiveBayesClassifier,
                       SimpleMarkovClassifier, DiscreteHopfieldClassifier)
from mdp.utils import weighted_choice

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
                
    def testSignumClassifier(self):
        c = SignumClassifier()
        res = c.classify(mdp.numx.array([[1, 2, -3, -4], [1, 2, 3, 4]]))
        assert c.input_dim == 4
        assert res.tolist() == [-1, 1]
        
    def testPerceptronClassifier(self):
        or_Classifier = PerceptronClassifier()
        for i in range(100):
            or_Classifier.train(mdp.numx.array([[0, 0]]), -1)
            or_Classifier.train(mdp.numx.array([[0, 1], [1, 0], [1, 1]]), 1)
        assert or_Classifier.input_dim == 2
        
        res = or_Classifier.classify(mdp.numx.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        assert res.tolist() == [-1, 1, 1, 1]
                            
        and_Classifier = PerceptronClassifier()
        for i in range(100):
            and_Classifier.train(mdp.numx.array([[0, 0], [0, 1], [1, 0]]), -1)
            and_Classifier.train(mdp.numx.array([[1, 1]]), 1)
        res = and_Classifier.classify(mdp.numx.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        assert res.tolist() == [-1, -1, -1, 1]
                            
        xor_Classifier = PerceptronClassifier()
        for i in range(100):
            xor_Classifier.train(mdp.numx.array([[0, 0], [1, 1]]), -1)
            xor_Classifier.train(mdp.numx.array([[0, 1], [1, 0]]), 1)
        res = xor_Classifier.classify(mdp.numx.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        assert res.tolist() != [-1, 1, 1, -1], "Something must be wrong here. XOR is impossible in a single-layered perceptron."
        
    def testNaiveBayesClassifier(self):
        ## This example needs some database and data in order to work
        # I did not want to include my spam mails with this code, so
        # it is here as an exercise for the reader. 
        pass
        
        #import string
        #import re
        #bc = NaiveBayesClassifier()
        #
        #regex = re.compile('[%s]' % re.escape(string.punctuation))
        #
        #SPAMTEXT_ = SPAMTEXT.split("\n\n")
        #NOSPAMTEXT_ = NOSPAMTEXT.split("\n\n")
        #
        #for SPAM in SPAMTEXT_:
        #    spam = regex.sub(' ', SPAM).lower().split()
        #    spamvals = mdp.numx.array([[hash(s) for s in spam]])
        #    bc.train(spamvals, -1)
        #   
        #for NOSPAM in NOSPAMTEXT_:
        #    nospam = regex.sub(' ', NOSPAM).lower().split()
        #    nospamvals = mdp.numx.array([[hash(s) for s in nospam]])
        #    bc.train(nospamvals, 1)
        #
        #text = regex.sub(' ', NOSPAMTEXT_[0]).lower().split()
        #textvals = mdp.numx.array([[hash(s) for s in text]])
        #print bc.prob(textvals)

    def testSimpleMarkovClassifier(self):
        mc = SimpleMarkovClassifier(dtype="unicode")
        text = "after the letter e follows either space or the letters r t or i"
        
        for word in text.split():
            word = word.lower()

            features = zip(" " + word)
            labels = list(word + " ")
            mc.train(mdp.numx.array(features), labels)
        
        assert mc.input_dim == 2
        
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
            assert numx.all(p == h.classify(numx.array([p])))
        
        for p in patterns:
            # check, if a noisy pattern is recreated
            noisy = numx.array(p)
            for i in range(len(noisy)):
                if numx.random.random() > 0.9:
                    noisy[i] = not noisy[i]
            retrieved = h.classify(numx.array([noisy]))
            # Hopfield nets are blind for inversion, need to check either case
            assert numx.all(retrieved == p) or numx.all(retrieved != p)
        

def get_suite(testname=None):
    return ClassifierTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

