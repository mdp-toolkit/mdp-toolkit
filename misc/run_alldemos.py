#!/usr/bin/env python
import os
import sys
import mdp
import new
import StringIO
import warnings

if len(sys.argv) == 1:
    warnings.filterwarnings("ignore",'.*',mdp.MDPWarning)
    fake_stdout = StringIO.StringIO()
    sys.stdout = fake_stdout

demodir = '../mdp/demo/'
err = sys.stderr

for demo in os.listdir(demodir):
    if demo[-7:].lower() == 'demo.py':
        novisual = new.module('novisual')
        print >> err, 'Running ' + demo
        try:
            if demo.lower() != 'checkpoint_demo.py':
                execfile(demodir+demo, novisual.__dict__)
            else:
                execfile(demodir+demo)
            print >> err, 'Done with ' + demo
        except Exception, e:
            print >> err, e
            print >> err, 'Failed!'
        print >> err, '--------'
