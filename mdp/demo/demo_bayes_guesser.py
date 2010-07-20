# -*- coding: utf-8 -*-
"""
Demo to check whether a text is more like the paragraphs in file one or more like file two.
Each sample set is separated from the next by two line breaks.

Based on Bayesian analysis – the same your spam filter uses.
Can be used to check whether one text belongs to one language or another – although
that is pretty easy to do for an algorithm. Just supply some paragraphs of text
from each language and let the classifier analyse them.
"""

import subprocess
import re
import string
import codecs
import operator
from optparse import OptionParser

import mdp
from mdp import ClassifierNode, numx
from mdp.utils import weighted_choice


class NaiveBayesClassifier(ClassifierNode):
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

    def _check_train_args(self, x, labels):
        if (isinstance(labels, (list, tuple, numx.ndarray)) and
            len(labels) != x.shape[0]):
            msg = ("The number of labels should be equal to the number of "
                   "datapoints (%d != %d)" % (len(labels), x.shape[0]))
            raise mdp.TrainingException(msg)

        if (not isinstance(labels, (list, tuple, numx.ndarray))):
            labels = [labels]

        if (not all(map(lambda x: abs(x) == 1, labels))):
            msg = "The labels must be either -1 or 1."
            raise mdp.TrainingException(msg)

    def _train(self, x, labels):
        """Update the internal structures according to the input data 'x'.

        x -- a matrix having different variables on different columns
             and observations on the rows.
        labels -- can be a list, tuple or array of labels (one for each data point)
              or a single label, in which case all input data is assigned to
              the same class.
        """
        for xi, labeli in mdp.utils.izip_stretched(x, labels):
            self._learn(xi, labeli)

        # clear input dim hack
        self._set_input_dim(None)

    def _learn(self, words, labels):
        # assume words is iterable of ints, labels is number
        if labels == -1:
            # mail is spam
            self.num_spam += 1
            for word in words:
                if word in self.spam:
                    self.spam[word] += 1
                else:
                    self.spam[word] = 1
        elif labels == 1:
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

    def _label(self, words):
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
        return - numx.sign(q - 1)



class LanguageGuesserDemo(object):
    def __init__(self, files):
        self.nbc = NaiveBayesClassifier(dtype="unicode")
        self.trainNaiveBayesClassifier(files)
        string = raw_input("Insert a string to check against both files:\n> ")
        print self.check_string(string)

    def trainNaiveBayesClassifier(self, files):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        def file_len(fname):
            f = open(fname)
            for i, l in enumerate(f):
                pass
            f.close()
            return i + 1

        dictfile = codecs.open(files[0], "r", "latin-1")
        print "Start learning from ‘%s’." % files[0]
        paragraph = []
        for num, line in mdp.utils.progressinfo(enumerate(dictfile), file_len(files[0])):
            # transform input to our needs
            #if num == 100: break
            line = line.strip("\n")
            words = regex.sub(' ', line).lower().strip().split()
            paragraph += words
            if not line and paragraph:
                self.nbc.train(mdp.numx.array([paragraph]), 1)
                paragraph = []
        if not line and paragraph:
            self.nbc.train(mdp.numx.array([paragraph]), 1)
        dictfile.close()

        dictfile = codecs.open(files[1], "r", "latin-1")
        print "Start learning from ‘%s’." % files[1]
        paragraph = []
        for num, line in mdp.utils.progressinfo(enumerate(dictfile), file_len(files[1])):
            # transform input to our needs
            #if num == 100: break
            line = line.strip("\n")
            words = regex.sub(' ', line).lower().strip().split()
            paragraph += words
            if not line and paragraph:
                self.nbc.train(mdp.numx.array([paragraph]), -1)
                paragraph = []
        if not line and paragraph:
            self.nbc.train(mdp.numx.array([paragraph]), -1)
        dictfile.close()

    def check_string(self, text):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub(' ', text).lower().split()
        textvals = mdp.numx.array([text])
        prob = self.nbc.prob(textvals)
        return {"first": prob[1], "second": prob[-1]}

if __name__ == '__main__':

    parser = OptionParser("usage: %prog --first=FILE --second=FILE [options] arg")
    parser.add_option("--first", dest="first",
                      help="select the first file to learn", metavar="FILE")
    parser.add_option("--second", dest="second",
                      help="select the second file to learn", metavar="FILE")

    (options, args) = parser.parse_args()

    if not (options.first and options.second):
        parser.error("You have to supply two files.")

    demo = LanguageGuesserDemo((options.first, options.second))
