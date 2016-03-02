from __future__ import print_function
# calls to os.system should be changed to subprocess.Popen!
# we don't need any temporary files and such. it is important
# to set the environment properly
# I call it like this:
# $ cd /home/tiziano/git/MDP/mdp-toolkit
# $ python testall.py /home/tiziano/python/x86_64/lib/pythonVERSION/site-packages

PARMS = {
         '3.4': ('numpy', None, 'joblib', 'scikits'),
         '3.5': ('numpy', None, 'joblib', 'scikits'),
         '2.7': ('scipy', None, 'parallel_python', 'libsvm', 'joblib', 'scikits'),
         }

import os
import sys
import subprocess

# get from sys.argv a directory to add to pythonpath
# /path/to/pythonVERSION/dir
if len(sys.argv) > 1:
    dirbase = sys.argv[1]
else:
    dirbase = '/dev/null'

# check that we are in our git repo
conds = (os.path.exists('.git'),
         os.path.basename(os.getcwd()) == 'mdp-toolkit',
         os.path.exists('mdp'),
         os.path.exists('bimdp'),
         )
if not all(conds):
    sys.stderr.write('Not in mdp git clone!')
    sys.exit(-1)

startwd = os.getcwd()
config = '-c "import mdp; import sys; sys.stdout.write(mdp.config.info())"'

# create command line
for vers in PARMS:
    print('Running: '+vers)
    path = dirbase.replace('VERSION', vers)
    os.chdir(startwd)
    wd = os.getcwd()
    env = {'MDPNUMX': PARMS[vers][0]}
    for dep in PARMS[vers][1:]:
        print('NoDep: '+str(dep))
        if dep is not None:
            key = 'MDP_DISABLE_'+dep.upper()
        else:
            key = 'MDP_DISABLE_NONE'
        env[key] = '1'
        for pack in ('mdp', 'bimdp'): 
            cmdline_base = ('MDPNUMX='+env['MDPNUMX'],
                            key+'=1',
                            'PYTHONPATH='+path+':'+wd,
                            ' /usr/bin/python'+vers,
                            '-m', 'pytest'
                            )

            cmdline_config = (config,)
            cmdline_tests = (
                             '--capture', 'fd',
                             '-x',
                             pack,
                             ' >',
                             '/tmp/mdp_current_test',
                             '2>&1',
                             )
            # show config
            #os.system(' '.join(cmdline_base+cmdline_config))
            sys.stdout.write('\n')
            # write out command line
            #print '  '+' '.join(cmdline_base+cmdline_tests)
            exit_status = os.system(' '.join(cmdline_base+cmdline_tests))
            if exit_status != 0:
                sys.stderr.write('='*30+' FAILURE '+'='*30)
                sys.stderr.write('\nLog is in /tmp/mdp_current_test.\n')
                sys.exit(-1)
