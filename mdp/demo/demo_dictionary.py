# -*- coding: utf-8 -*-
"""
Demo to show how mdp can help with creating a simple word generator based on
Markovian assumptions.

This demo assumes locate is installed and some OpenOffice dictionaries are available.
"""

import subprocess
import re
import string
import codecs

import mdp
from mdp.nodes import SimpleMarkovClassifier
from mdp.utils import weighted_choice

class DictionaryDemo():
    """This demo generates words from a selected dictionary by calculating the transition
probabilities from two consecutive letters to the next.
    """
    def __init__(self, num_words, correlation):
        print self.__doc__
        
        self.correlation = correlation
        self.mc = SimpleMarkovClassifier(dtype="unicode")
        
        dict = self.show_available()
        if dict is None:
            return
        self.trainSimpleMarkovClassifier(dict)
        self.print_transition_probabilities()
        self.get_words(num_words)
    
    def show_available(self):
        available_dicts = []
        for name in ('/usr/share/dict/words', ):
            available_dicts.append(name)

        try:
            possible_dicts = subprocess.Popen(["locate", ".dic"], stdout=subprocess.PIPE).stdout
        except OSError:
            raise NotImplementedError("Sorry, you need to have ‘locate’ on your system.")
        print possible_dicts
        filename = re.compile('.*\/\w{2,3}_\w{2,3}\.dic$')
        
        for name in possible_dicts:
            match = filename.match(name)
            if match:
                available_dicts.append(name.strip("\n"))
        for i, dict in enumerate(available_dicts):
            print (" %2i) %s" % (i + 1, dict))
        num = raw_input("Which dictionary do you want to use? ")
        
        try:
            return available_dicts[int(num) - 1]
        except IndexError:
            return None
        
    def trainSimpleMarkovClassifier(self, dictionary):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        dictfile = codecs.open(dictionary, "r", "latin-1")

        def file_len(fname):
            f = open(fname)
            for i, l in enumerate(f):
                pass
            f.close()
            return i + 1
        
        print "Start learning from ‘%s’." % dictionary
        for num, word in mdp.utils.progressinfo(enumerate(dictfile), file_len(dictionary)):
            # transform input to our needs
            #if num == 100: break
            
            # remove punctuation
            word = regex.sub(' ', word).lower().strip().split()
            word = word[0]

            shifted_words = [" " * i + word for i in range(self.correlation, 0, -1)]
            words = zip(*shifted_words)
            labels = list(word + " ")
            self.mc.train(mdp.numx.array(words), labels)
        
        dictfile.close()
        
    def print_transition_probabilities(self):
        features = self.mc.features
        for feature, count in features.items():
            if count:
                prob = self.mc.prob(mdp.numx.array([feature]))
                for p in prob:
                    for k, v in p.items():
                        if v:
                            print "".join(feature).replace(" ", "_"), "->", k.replace(" ", "_"), "(", v, ")"
    
    def get_words(self, num_words):
        for _ in range(num_words):
            features = [" "] * (self.correlation)
            for __ in range(50): # have a maximum length
                f = mdp.numx.array([features[-self.correlation:]])
                new_f = weighted_choice(self.mc.prob(f)[0], True)
                if new_f is None:
                    break
                features.append(new_f)
            print "".join(features)

if __name__ == '__main__':
    demo = DictionaryDemo(20, 2)

