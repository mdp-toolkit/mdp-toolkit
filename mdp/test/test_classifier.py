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
from mdp.nodes import SignumClassifier, PerceptronClassifier, NaiveBayesClassifier

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
        assert res.tolist() == [-1, 1]
        
    def testPerceptronClassifier(self):
        or_Classifier = PerceptronClassifier()
        for i in range(100):
            or_Classifier.train(mdp.numx.array([[0, 0]]), -1)
            or_Classifier.train(mdp.numx.array([[0, 1], [1, 0], [1, 1]]), 1)
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


def get_suite(testname=None):
    return ClassifierTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

