"""
This module contains the basic classes for job execution via a scheduler.
"""

import thread
import time


class Job(object):
    """Base class for jobs.
    
    Note that a function can be used as a job as well.
    """
    
    def __call__(self):
        """Do the job and return the result."""
        return None
    
    
class TestJob(Job):
    """Simple test job to play around."""

    def __init__(self, x):
        self.x = x
      
    def __call__(self):
        return self.x**2


class ResultContainer(object):
    """Abstract base class for result containers."""
    
    def add_result(self, result):
        """Store a result in the container."""
        pass
    
    def get_results(self):
        """Return results and reset container."""
        pass
    
class ListResultContainer(ResultContainer):
    """Default result container.
    
    Simply wraps a list. Note that the order of the results may be different
    from the order of the jobs, since jobs may finish sooner or later than
    other jobs. If the order is important then use the order module.
    """
    
    def __init__(self):
        super(ListResultContainer, self).__init__()
        self._results = []
        
    def add_result(self, result):
        self._results.append(result)
        
    def get_results(self):
        results = self._results
        self._results = []
        return results


class Scheduler(object):
    """Base class and trivial implementation for schedulers.
    
    New jobs are added with add_task(job) where job, is an object with a 
    __call__() function or simply a function.
    get_results then returns the result container (and locks if jobs are
    pending).
    
    In this simple scheduler implementation the jobs are simply executed in the 
    add_job method.
    """

    def __init__(self, result_container=ListResultContainer(), verbose=False):
        """Initialize the scheduler.
        
        result_container -- Instance of ResultContainer that is used to store
            the results (default is ListResultContainer).
        verbose -- If True then status messages will be printed to sys.stdout.
        """
        self.result_container = result_container
        self.verbose = verbose
        self.n_jobs_running = 0  # number of jobs that are currently running
        self.n_jobs_finished = 0  # count the number of all finished jobs
        self.lock = thread.allocate()  # general lock for this class
    
    def add_job(self, job):
        """Add a job to be executed.
        
        job -- Executable which returns a result.
        
        In this simple implementation the job is simply executed. The method
        blocks if another job is already running (which may happen if multiple 
        threads are used). 
        
        This method is overridden in more complex schedulers. Generally this
        method is potentially blocking (e.g. when the job queue is full).
        """
        self.lock.acquire()
        self.n_jobs_running += 1
        result = job()
        self.n_jobs_running -= 1
        self.lock.release()
        self._store_result(result)
        if self.verbose:
            print "    finished job no. %d" % self.n_jobs_finished
        
    def _store_result(self, result):
        """Store a result in the internal result container."""
        self.lock.acquire()
        self.result_container.add_result(result)
        self.n_jobs_finished += 1
        self.lock.release()
    
    def get_results(self):
        """Get the accumulated results from the result container.
        
        This function blocks if there are open jobs. 
        """
        while True:
            self.lock.acquire()
            if self.n_jobs_running == 0:
                results = self.result_container.get_results()
                self.lock.release()
                return results
            else:
                self.lock.release();
                time.sleep(1); 

    def cleanup(self):
        """Controlled shutdown.
        
        Should be called when the scheduler is no longer needed (e.g. to shut
        down slave processes and such).
        """
        pass
    
                    
    
