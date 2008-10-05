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
    """Scheduler that distributes the job to multiple processes.
    
    The subprocess module is used to start the requested number of processes.
    The execution of each job is internally managed by dedicated thread.
    
    This scheduler should work on all platforms (at least on Linux,
    Windows XP and Vista). 
    """
    
    def __init__(self, result_container=scheduling.ListResultContainer(), 
                 verbose=False, n_processes=1,
                 source_paths=None):
        """Initialize the scheduler and start the slave processes.
        
        result_container -- ResultContainer used to store the results.
        verbose -- If  True to get progress reports from the scheduler.
        n_processes -- Number of processes used in parallel. This should
            correspond to the number of processors / cores.
        source_paths -- List of paths to the source code of the project using 
            the scheduler. These paths will be appended to sys.path in the
            processes to make the job unpickling work. 
            A single path instead of a list is also accepted.
            Set to None if no sources are needed for unpickling the job (this is
            the default value).
        """
        scheduling.Scheduler.__init__(self, result_container=result_container,
                                      verbose=verbose)
        self.n_processes = n_processes
        # get the location of this module to start the processes
        module_path = os.path.dirname(inspect.getfile(sys._getframe(0)))
        module_path = os.path.abspath(module_path)
        module_file = os.path.join(module_path, "process_schedule.py")
        # Note: -u argument is important on Windows to set stdout to binary 
        #    mode. Otherwise you might get a strange error message for 
        #    copy_reg.
        process_args = ["python", "-u", module_file]
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
        
        If a process is still running a job an exception is raised.
        """
        self.lock.acquire()
        if len(self._free_processes) < self.n_processes:
            raise Exception("Some slave process is still working.")
        for process in self._free_processes:
            pickle.dump("EXIT", process.stdin) 
        self.lock.release()
        
    def add_job(self, job):
        """Add a job, if possible without blocking.
        
        It blocks when the system is not able to start a new thread
        or when the processes are all in use.
        """
        job_started = False
        while not job_started:
            self.lock.acquire()
            if not len(self._free_processes):
                # release lock for other threads and wait
                self.lock.release()
                time.sleep(0.5)
            else:
                self.n_jobs_running += 1
                try:
                    process = self._free_processes.pop()
                    self.lock.release()
                    thread.start_new(self._job_thread, (job, process))
                    job_started = True
                except thread.error:
                    if self.verbose:
                        print ("unable to create new job thread," 
                               " waiting 2 seconds...")
                    time.sleep(2)
                    
    def _job_thread(self, job, process): 
        """Thread function which cares for a single job.
        
        It picks a free process, pushes the to it via stdin, waits for the
        result on stdout, passes the result to the result container, frees
        the process and then exits. 
        """
        try:
            # push the job to the process
            pickle.dump(job, process.stdin)
            # wait for result to arrive
            result = pickle.load(process.stdout)
        except:
            print "\nFAILED TO EXECUTE JOB IN PROCESS!\n"
            sys.exit()
        # store the result and clean up
        self._store_result(result)
        self._free_processes.append(process)
        self.n_jobs_running -= 1


def _process_run():
    """Run this function in a worker process to receive and run jobs.
    
    It waits for jobs on stdin, and sends the results back via stdout.
    """
    # use sys.stdout only for pickled objects, everything else goes to stderr
    pickle_out = sys.stdout
    # TODO: add process identifier prefix?
    sys.stdout = sys.stderr
    exit_loop = False
    while not exit_loop:
        try:
            # wait for job to arrive
            job = pickle.load(sys.stdin)
            if job == "EXIT":
                exit_loop = True
            else:
                result = job()
                pickle.dump(result, pickle_out)
                pickle_out.flush()
        except Exception, exception:
            # return the exception instead of the result
            print "\nJOB CAUSED EXCEPTION IN PROCESS:\n"
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
    
    
    