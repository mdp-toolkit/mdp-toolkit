"""
Process based scheduler for distribution across multiple CPU cores.
"""

import sys
import os
import cPickle as pickle
import thread
import subprocess
import time
import inspect
import traceback

import scheduling


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
        module_path = os.path.dirname(inspect.getfile(sys._getframe(0)))
        module_path = os.path.abspath(module_path)
        module_file = os.path.join(module_path, "process_schedule.py")
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
        
    def cleanup(self):
        """Shut down the slave processes.
        
        If a process is still running a task an exception is raised.
        """
        self.lock.acquire()
        if len(self._free_processes) < self.n_processes:
            raise Exception("Some slave process is still working.")
        for process in self._free_processes:
            pickle.dump("EXIT", process.stdin) 
        self.lock.release()
        
    def add_task(self, data, task_callable=None):
        """Add a task, if possible without blocking.
        
        It blocks when the system is not able to start a new thread
        or when the processes are all in use.
        """
        task_started = False
        while not task_started:
            self.lock.acquire()
            if not len(self._free_processes):
                # release lock for other threads and wait
                self.lock.release()
                time.sleep(0.5)
            else:
                self.n_tasks_running += 1
                try:
                    process = self._free_processes.pop()
                    self.lock.release()
                    thread.start_new(self._task_thread, (task, process))
                    task_started = True
                except thread.error:
                    if self.verbose:
                        print ("unable to create new task thread," 
                               " waiting 2 seconds...")
                    time.sleep(2)
                    
    def _task_thread(self, task, process): 
        """Thread function which cares for a single task.
        
        It picks a free process, pushes the to it via stdin, waits for the
        result on stdout, passes the result to the result container, frees
        the process and then exits. 
        """
        try:
            # push the task to the process
            pickle.dump(task, process.stdin)
            # wait for result to arrive
            result = pickle.load(process.stdout)
        except:
            print "\nFailed to execute task in process!\n"
            sys.exit()
        # store the result and clean up
        self._store_result(result)
        self._free_processes.append(process)
        self.n_tasks_running -= 1


def _process_run():
    """Run this function in a worker process to receive and run tasks.
    
    It waits for tasks on stdin, and sends the results back via stdout.
    """
    # use sys.stdout only for pickled objects, everything else goes to stderr
    pickle_out = sys.stdout
    # TODO: add process identifier prefix?
    sys.stdout = sys.stderr
    exit_loop = False
    while not exit_loop:
        try:
            # wait for task to arrive
            task = pickle.load(sys.stdin)
            if task == "EXIT":
                exit_loop = True
            else:
                result = task()
                pickle.dump(result, pickle_out)
                pickle_out.flush()
        except Exception, exception:
            # return the exception instead of the result
            print "\ntask caused exception in process:\n"
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
    
    
    