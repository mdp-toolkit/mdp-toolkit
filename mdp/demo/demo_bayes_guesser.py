# -*- coding: utf-8 -*-
"""
Demo to check whether a text is more like the paragraphs in file one or more like file two.
Each sample set is separated from the next by two line breaks.

Based on Bayesian analysis – the same your spam filter uses.
Can be used to check whether one text belongs to one language or another – although
that is pretty easy to do for an algorithm. Just supply some paragraphs of text
from each language and let the classifier analyse them.
"""

import mdp

# -*- coding: utf-8 -*-
"""
Demo to show how mdp can help with creating a simple word generator based on
Markovian assumptions.
"""

import subprocess
import re
import string
import codecs
from optparse import OptionParser

import mdp
from mdp.nodes import NaiveBayesClassifier
from mdp.utils import weighted_choice


class LanguageGuesserDemo():
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
        
        

parser = OptionParser("usage: %prog --first=FILE --second=file [options] arg")
parser.add_option("--first", dest="first",
                  help="select the first file to learn", metavar="FILE")
parser.add_option("--second", dest="second",
                  help="select the second file to learn", metavar="FILE")

(options, args) = parser.parse_args()

if not (options.first and options.second):
    parser.error("You have to supply first and second.")
    
demo = LanguageGuesserDemo((options.first, options.second))

