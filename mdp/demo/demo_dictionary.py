# -*- coding: utf-8 -*-
"""
Demo to show how mdp can help with creating a simple word generator based on
Markovian assumptions.

This demo assumes locate is installed and some OpenOffice dictionaries are available.
"""

import os
import subprocess
import re
import string
import codecs

from optparse import OptionParser

import mdp
from mdp.nodes import SimpleMarkovClassifier
from mdp.utils import weighted_choice

def show_available_dicts():
    available_dicts = []
    for name in ('/usr/share/dict/words', ):
        try:
            size = os.path.getsize(name)
            available_dicts.append((name, size))
        except OSError:
            pass

    try:
        possible_dicts = subprocess.Popen(["locate", ".dic"],
                                          stdout=subprocess.PIPE).stdout
    except OSError:
        raise NotImplementedError("Sorry, you need to have ‘locate’ on your system.")

    filename = re.compile('.*\/\w{2,3}_\w{2,3}\.dic$')

    for name in possible_dicts:
        match = filename.match(name)
        if match:
            f = name.strip("\n")
            try:
                size = os.path.getsize(f)
                available_dicts.append((name.strip("\n"), size))
            except OSError:
                pass
    for i, dict_ in enumerate(available_dicts):
        print (" %2i) %s [%i kB]" % (i + 1, dict_[0], dict_[1] / 1024))
    num = raw_input("Which dictionary do you want to use? ")

    try:
        return available_dicts[int(num) - 1][0]
    except IndexError:
        return None



class DictionaryDemo(object):
    """This demo generates words from a selected dictionary by calculating
    the transition probabilities from two consecutive letters to the next.
    """
    def __init__(self, dictionary, correlation, verbose=False):
        self._correlation = correlation
        self._dictionary = dictionary
        self._verbose = verbose

        if self._verbose:
            print self.__doc__

        self.mc = SimpleMarkovClassifier(dtype="unicode")

        self.trainSimpleMarkovClassifier()
        if self._verbose:
            self.print_transition_probabilities()

    def trainSimpleMarkovClassifier(self):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        dictfile = codecs.open(self._dictionary, "r", "latin-1")

        def file_len(fname):
            f = open(fname)
            for i, l in enumerate(f):
                pass
            f.close()
            return i + 1

        if self._verbose:
            print "Start learning from ‘%s’." % self._dictionary
        for num, word in mdp.utils.progressinfo(enumerate(dictfile),
                                                file_len(self._dictionary)):
            # transform input to our needs
            #if num == 100: break

            # remove punctuation
            word = regex.sub(' ', word).lower().strip().split()
            try:
                word = word[0]
            except IndexError:
                continue

            shifted_words = [" " * i + word for i in range(self._correlation, 0, -1)]
            words = zip(*shifted_words)
            labels = list(word + " ")
            self.mc.train(mdp.numx.array(words), labels)

        dictfile.close()

    def print_transition_probabilities(self):
        print "Transition probabilities:"
        features = self.mc.features
        for feature, count in features.items():
            if count:
                prob = self.mc.prob(mdp.numx.array([feature]))
                for p in prob:
                    for k, v in p.items():
                        if v:
                            print "".join(feature).replace(" ", "_"), \
                                  "->", k.replace(" ", "_"), \
                                  "(", ("%7.3f %%" % (v * 100)), ")"

    def get_words(self, num_words):
        for _ in range(num_words):
            features = [" "] * (self._correlation)
            for __ in range(50): # have a maximum length
                f = mdp.numx.array([features[-self._correlation:]])
                new_f = weighted_choice(self.mc.prob(f)[0], True)
                if new_f is None:
                    break
                features.append(new_f)
            print "".join(features)

if __name__ == '__main__':
    parser = OptionParser("usage: %prog [options] dictionary")

    parser.add_option("-c", "--correlation",
                      help="The amount of correlation between the letters.",
                      default=2, type=int)
    parser.add_option("-n", "--number", dest="num_words",
                      help="The number of examples.", default=20, type=int)
    parser.add_option("-q", "--quiet", dest="verbose", action="store_false",
                      help=u"Don’t give any statistical information.",
                      default=True)
    parser.epilog = "If no dictionary is given, a list of possible dictionaries is produced."

    (options, args) = parser.parse_args()

    if len(args) > 1:
        parser.error("Only one dictionary file is supported.")

    if not args:
        dictionary = show_available_dicts()
    else:
        dictionary = args[0]

    if dictionary is None:
        exit

    demo = DictionaryDemo(dictionary, options.correlation, options.verbose)
    demo.get_words(options.num_words)
