PARMS = {'2.5': ('numpy',),
         '2.6': ('scipy', 'parallel_python', 'shogun', 'libsvm', 'joblib'),
         '2.7': ('numpy',),
         '3.1': ('numpy',),
         }

import os
import sys
import subprocess
import time

# check that we are in our git repo
conds = (os.path.exists('.git'),
         os.path.basename(os.getcwd()) == 'mdp-toolkit',
         os.path.exists('mdp'),
         os.path.exists('bimdp'),
         )
if not all(conds):
    sys.stderr.write('Not in mdp git clone!')
    sys.exit(-1)

null = open('/tmp/mdp_current_test', 'w')

# create command line
for vers in PARMS:
    env = {'MDPNUMX': PARMS[vers][0]}
    for dep in PARMS[vers][1:]:
        key = 'MDP_DISABLE_'+dep.upper()
        env[key] = '1'
        cmdline_base = ('export MDPNUMX='+env['MDPNUMX'],
                        key+'=1;',
                        'python'+vers,
                        )

        cmdline_config = ('-c "import mdp; import sys; sys.stdout.write(mdp.config.info())"',)
        cmdline_tests = (os.path.join('mdp','test','run_tests.py'),
                         '--capture', 'fd',
                         '-x',
                         'mdp',
                         'bimdp',
                         '>',
                         '/tmp/mdp_current_test',
                         '2>&1',
                         )
        # show config
        print 'Running: '
        os.system(' '.join(cmdline_base+cmdline_config))
        sys.stdout.write('\n')
        #print 'Running: ', ' '.join(cmdline)
        exit_status = os.system(' '.join(cmdline_base+cmdline_tests))
        if exit_status != 0:
            sys.stderr.write('='*30+' FAILURE '+'='*30)
            sys.exit(-1)
