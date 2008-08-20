"""Process based scheduler for distribution across multiple CPU cores."""

import sys
import os
import cPickle
import thread
import subprocess
import time
import inspect
import traceback

import scheduling


def get_module_path(filename=None, frame=1):
    """Return the absolute path of the module this function is called from.
    
    The returned path has the os specific format.
    
    filename -- If specified the path is joined with it.
    frame -- Specifies for which stage in the call stack the path is returned,
        so 1 gives the calling module, 2 the module that called the module 
        and so on.
    """
    path = os.path.dirname(inspect.getfile(sys._getframe(frame)))
    if filename == None:
        return path
    else:
        return os.path.join(path, filename)


class ProcessScheduler(scheduling.Scheduler):
    """Scheduler that distributes the job to multiple processes.
    
    The subprocess module is used to start the requested number of processes.
    The execution of each job is internally managed by dedicated thread.
    """
    
    def __init__(self, code_path=None, 
                 result_container=scheduling.ListResultContainer(), 
                 verbose=False, n_processes=1):
        """Initialize the scheduler and start the slave processes.
        
        code_path -- Path to the source code of the project using the scheduler.
            Can also be None if no sources are needed for unpickling the job.
        result_container -- ResultContainer used to store the results.
        verbose -- If  True to get progress reports from the scheduler.
        n_processes -- Number of processes used in parallel. This should
            correspond to the number of processors / cores.
        """
        scheduling.Scheduler.__init__(self, result_container=result_container,
                                     verbose=verbose)
        self.n_processes = n_processes
        # list of processes not in use, start the processes now
        module_path = os.path.abspath(get_module_path())
        module_path = os.path.join(module_path, "process_schedule.py")
        # TODO: set current working directory, cwd="..."?
        # Note: -u argument is important on Windows 
        #    to set stdout to binary mode. Otherwise you might get a
        #    strange error message regarding copy_reg.
        process_args = ("python", "-u", module_path)
        if code_path is not None:
            process_args += (code_path,)
        self.free_processes = [subprocess.Popen(args=process_args,
                                                stdout=subprocess.PIPE, 
                                                stdin=subprocess.PIPE)
                               for i in range(self.n_processes)]
        
    def cleanup(self):
        """Shut down the slave processes."""
        self.lock.acquire()
        if len(self.free_processes) < self.n_processes:
            raise Exception("Some slave process is still working.")
        for process in self.free_processes:
            cPickle.dump("EXIT", process.stdin) 
        self.lock.release()
        
    
    def add_job(self, job):
        """Add a job, if possible without blocking.
        
        It blocks when the system is not able to start a new thread
        or when the processes are all in use.
        """
        job_started = False
        while not job_started:
            self.lock.acquire()
            if not len(self.free_processes):
                # block and wait for a free process
                # release lock for other threads
                self.lock.release()
                time.sleep(0.5)
            else:
                self.n_jobs_running += 1
                try:
                    process = self.free_processes.pop()
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
            # push the job the process
            cPickle.dump(job, process.stdin)
            # wait for result to arrive
            result = cPickle.load(process.stdout)
        except:
            print "\nFAILED TO EXECUTE JOB IN PROCESS!\n"
            sys.exit()
        # store the result and clean up
        self.lock.acquire()
        self._store_result(result)
        self.free_processes.append(process)
        self.n_jobs_finished += 1
        self.n_jobs_running -= 1
        if self.verbose:
            print "    finished job no. %d" % self.n_jobs_finished
        self.lock.release()


def process_execute():
    """This is the process execute method."""
    # use sys.stdout only for pickled objects, everything else goes to stderr
    pickle_out = sys.stdout
    # TODO: add process identifier prefix?
    sys.stdout = sys.stderr
    exit_loop = False
    while not exit_loop:
        try:
            # wait for job to arrive
            job = cPickle.load(sys.stdin)
            if job == "EXIT":
                exit_loop = True
            else:
                result = job()
                cPickle.dump(result, pickle_out)
                pickle_out.flush()
        except Exception, exception:
            # return the exception instead of the result
            print "\nJOB CAUSED EXCEPTION IN PROCESS:\n"
            print exception
            traceback.print_exc()
            sys.stdout.flush()
            sys.exit()
        
                    
if __name__ == "__main__":
    # append MDP source path 
    module_file = os.path.abspath(inspect.getfile(sys._getframe(0)))
    module_path = os.path.dirname(module_file)
    sys.path.append(os.path.join(module_path.split("mdp")[0])[0:-1])
    # append project source path, which is the first argument
    if len(sys.argv) > 1:
        sys.path.append(sys.argv[1])
    process_execute()
    
    
    