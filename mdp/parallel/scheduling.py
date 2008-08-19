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
        self.results = []
        
    def add_result(self, result):
        self.results.append(result)
        
    def get_results(self):
        results = self.results
        self.results = []
        return results


class Scheduler(object):
    """Abstract base class for schedulers.
    
    New jobs are added with add_task(job) where job, is an object with a 
    __call__() function or simply a function.
    
    get_results then returns the result container (and locks if jobs are
    pending).
    """

    def __init__(self, result_container=ListResultContainer(), verbose=False):
        """Initialize the scheduler."""
        self.result_container = result_container
        self.verbose = verbose
        self.n_jobs_running = 0
        self.n_jobs_finished = 0  # count the jobs of all finished jobs
        # this lock can be used by subthreads responsible for job allocation
        self.lock = thread.allocate()
    
    def add_job(self, job):
        """Add a job to be executed.
        
        Depending on the scheduler state this function is non-blocking or
        blocking. One reason for blocking can be a full job-queue.
        """
        pass
    
    def _store_result(self, result):
        """Store result in the result container."""
        self.result_container.add_result(result)
    
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
        """Controlled shutdown, should be called at the end."""
        pass
    
    
class SimpleScheduler(Scheduler):
    """Most simple scheduler implementation.
    
    The jobs are simply executed in the add_job function.
    """
    
    def add_job(self, job):
        """Execute job directly and blocks until job is done."""
        self.lock.acquire()
        self._store_result(job())
        self.n_jobs_finished += 1
        self.lock.release()
            
 
                    
    
