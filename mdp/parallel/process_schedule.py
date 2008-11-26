"""
Process based scheduler for distribution across multiple CPU cores.
"""



import sys
import os
import cPickle as pickle
import thread
import subprocess
import time
import traceback
import warnings

if __name__ == "__main__":
    # shut off warnings of any kinds
    warnings.filterwarnings("ignore", ".*")
    # trick to find mdp in a remote process
    mdp_path = __file__.split("mdp")[0]
    sys.path.append(mdp_path)


import mdp
import scheduling

# TODO: implement caching of callable in process?

class ProcessScheduler(scheduling.Scheduler):
    """Scheduler that distributes the task to multiple processes.
    
    The subprocess module is used to start the requested number of processes.
    The execution of each task is internally managed by dedicated thread.
    
    This scheduler should work on all platforms (at least on Linux,
    Windows XP and Vista). 
    """
    
    def __init__(self, result_container=None, 
                 verbose=False, n_processes=1,
                 source_paths=None, python_executable=None):
        """Initialize the scheduler and start the slave processes.
        
        result_container -- ResultContainer used to store the results.
        verbose -- If  True to get progress reports from the scheduler.
        n_processes -- Number of processes used in parallel. This should
            correspond to the number of processors / cores.
        source_paths -- List of paths to the source code of the project using 
            the scheduler. These paths will be appended to sys.path in the
            processes to make the task unpickling work. 
            A single path instead of a list is also accepted.
            Set to None if no sources are needed for unpickling the task (this is
            the default value).
        python_executable -- Python executable that is used for the processes.
            The default value is None, in which case sys.executable will be
            used.
        """
        scheduling.Scheduler.__init__(self, result_container=result_container,
                                      verbose=verbose)
        self.n_processes = n_processes
        if python_executable is None:
            python_executable = sys.executable
        # get the location of this module to start the processes
        # old way of getting it:
        # module_path = os.path.dirname(inspect.getfile(sys._getframe(0)))
        # module_path = os.path.abspath(module_path)
        # better new way (requires mdp import at the top):
        module_path = os.path.dirname(mdp.__file__)
        module_file = os.path.join(module_path, "parallel", 
                                   "process_schedule.py")
        # Note: -u argument is important on Windows to set stdout to binary 
        #    mode. Otherwise you might get a strange error message for 
        #    copy_reg.
        process_args = [python_executable, "-u", module_file]
        if type(source_paths) is str:
            source_paths = [source_paths]
        if source_paths is not None:
            process_args += source_paths
        # list of processes not in use, start the processes now
        self._free_processes = [subprocess.Popen(args=process_args,
                                                stdout=subprocess.PIPE, 
                                                stdin=subprocess.PIPE)
                                for _ in range(self.n_processes)]
        
    def shutdown(self):
        """Shut down the slave processes.
        
        If a process is still running a task then an exception is raised.
        """
        self.lock.acquire()
        if len(self._free_processes) < self.n_processes:
            raise Exception("some slave process is still working")
        for process in self._free_processes:
            pickle.dump("EXIT", process.stdin) 
        self.lock.release()
        
    def _process_task(self, data, task_callable, task_index):
        """Add a task, if possible without blocking.
        
        It blocks when the system is not able to start a new thread
        or when the processes are all in use.
        """
        if self.copy_callable:
            task_callable = task_callable.copy()
        task_started = False
        while not task_started:
            if not len(self._free_processes):
                # release lock for other threads and wait
                self.lock.release()
                time.sleep(1.5)
                self.lock.acquire()
            else:
                try:
                    process = self._free_processes.pop()
                    self.lock.release()
                    thread.start_new(self._task_thread, 
                                     (process, data, task_callable, task_index))
                    task_started = True
                except thread.error:
                    if self.verbose:
                        print ("unable to create new task thread," 
                               " waiting 2 seconds...")
                    time.sleep(2)
                    
    def _task_thread(self, process, data, task_callable, task_index): 
        """Thread function which cares for a single task.
        
        The task is pushed to the process via stdin, then we wait for the
        result on stdout, pass the result to the result container, free
        the process and exit. 
        """
        try:
            # push the task to the process
            pickle.dump((data, task_callable, task_index), process.stdin,
                        protocol=-1)
            # wait for result to arrive
            result = pickle.load(process.stdout)
        except:
            traceback.print_exc()
            self._free_processes.append(process)
            sys.exit("failed to execute task %d in process:" % task_index)
        # store the result and clean up
        self._store_result(result, task_index)
        self._free_processes.append(process)


def _process_run():
    """Run this function in a worker process to receive and run tasks.
    
    It waits for tasks on stdin, and sends the results back via stdout.
    """
    # use sys.stdout only for pickled objects, everything else goes to stderr
    pickle_out = sys.stdout
    sys.stdout = sys.stderr
    exit_loop = False
    while not exit_loop:
        task = None
        try:
            # wait for task to arrive
            task = pickle.load(sys.stdin)
            if task == "EXIT":
                exit_loop = True
            else:
                result = task[1](task[0])
                pickle.dump(result, pickle_out, protocol=-1)
                pickle_out.flush()
        except Exception, exception:
            # return the exception instead of the result
            if task is None:
                print "unpickling a task caused an exception in a process:"
            else:
                print "task %d caused exception in process:" % task[2]
            print exception
            traceback.print_exc()
            sys.stdout.flush()
            sys.exit()
        
                    
if __name__ == "__main__":
    # all arguments are expected to be code paths to be appended to sys.path
    if len(sys.argv) > 1:
        for sys_arg in sys.argv[1:]:
            sys.path.append(sys_arg)
    _process_run()
    
    
    
