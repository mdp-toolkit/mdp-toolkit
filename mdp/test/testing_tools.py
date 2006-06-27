## Automatically adapted for numpy Jun 26, 2006 by 

"""Tools for the test- and benchmark functions."""

import time
import mdp
numx = mdp.numx
from mdp.utils import assert_array_equal, assert_array_almost_equal, \
     assert_equal, assert_almost_equal, iscomplexobj

#### test tools
def assert_array_almost_equal_diff(x,y,digits,err_msg=''):
    x,y = numx.asarray(x), numx.asarray(y)
    msg = '\nArrays are not almost equal'
    assert 0 in [len(numx.shape(x)),len(numx.shape(y))] \
           or (len(numx.shape(x))==len(numx.shape(y)) and \
               numx.alltrue(numx.equal(numx.shape(x),numx.shape(y)))),\
               msg + ' (shapes %s, %s mismatch):\n\t' \
               % (numx.shape(x),numx.shape(y)) + err_msg
    maxdiff = max(numx.ravel(abs(x-y)))/\
              max(max(abs(numx.ravel(x))),max(abs(numx.ravel(y)))) 
    if iscomplexobj(x) or iscomplexobj(y): maxdiff = maxdiff/2
    cond =  maxdiff< 10**(-digits)
    msg = msg+'\n\t Relative maximum difference: %e'%(maxdiff)+'\n\t'+\
          'Array1: '+str(x)+'\n\t'+\
          'Array2: '+str(y)+'\n\t'+\
          'Absolute Difference: '+str(abs(y-x))
    assert cond, msg 

#### benchmark tools

# function used to measure time
TIMEFUNC = time.time

def timeit(func,*args,**kwargs):
    """Return function execution time in 1/100ths of a second."""
    tstart = TIMEFUNC()
    func(*args,**kwargs)
    return (TIMEFUNC()-tstart)*100.

def run_benchmarks(bench_funcs, time_digits=15):
    results_str = '| %%s | %%%d.2f |' % time_digits
    label_str = '| %%s | %s |' % 'Time (sec/100)'.center(time_digits)
    
    # loop over all benchmarks functions
    for func, args_list in bench_funcs:
        # number of combinations of arguments(cases)
        ncases = len(args_list)
        funcname = func.__name__[:-10]

        # loop over all cases
        for i in range(ncases):
            args = args_list[i]
            # execute function
            t = timeit(func, *args)
            # format description string
            descr = funcname + str(tuple(args))

            if i==0:
                # print summary table header
                descrlen = len(descr)+6
                results_strlen = time_digits+descrlen+7
                print '\n----> Timing results (%s, %d cases):' % (funcname, ncases)
                print func.__doc__
                print '+'+'-'*(results_strlen-2)+'+'
                print label_str % 'Description'.center(descrlen)
                print '+'+'-'*(results_strlen-2)+'+'        
            # print summary table entry
            print results_str % (descr.center(descrlen), t)

        # print summary table tail
        print '+'+'-'*(results_strlen-2)+'+'
