"""
Thread based scheduler for distribution across multiple CPU cores.
"""

import threading
import time

from scheduling import Scheduler, cpu_count

SLEEP_TIME = 0.1  # time spend sleeping when waiting for a thread to finish


class ThreadScheduler(Scheduler):
    """Thread based scheduler.
    
    Because of the GIL this only makes sense if most of the time is spend in
    numpy calculations (or some other external non-blocking C code) or for IO,
    but can be more efficient than ProcessScheduler because of the
    shared memory.
    """

    def __init__(self, result_container=None, verbose=False, n_threads=1):
        """Initialize the scheduler.
        
        result_container -- ResultContainer used to store the results.
        verbose -- Set to True to get progress reports from the scheduler
            (default value is False).
        n_threads -- Number of threads used in parallel. If None (default)
            then the number of detected CPU cores is used.
        """
        super(ThreadScheduler, self).__init__(
                                            result_container=result_container,
                                            verbose=verbose)
        if n_threads:
            self._n_threads = n_threads
        else:
            self._n_threads = cpu_count()
        self._n_active_threads = 0
        
    def _process_task(self, data, task_callable, task_index):
        """Add a task, if possible without blocking.
        
        It blocks when the maximum number of threads is reached (given by
        n_threads) or when the system is not able to start a new thread.
        """
        task_started = False
        while not task_started:
            if not self._n_threads - self._n_active_threads:
                # release lock for other threads and wait
                self._lock.release()
                time.sleep(SLEEP_TIME)
                self._lock.acquire()
            else:
                try:
                    self._lock.release()
                    task_callable = task_callable.fork()
                    thread = threading.Thread(target=self._task_thread,
                                              args=(data, task_callable,
                                                    task_index))
                    thread.start()
                    task_started = True
                except thread.error:
                    if self.verbose:
                        print ("unable to create new thread," 
                               " waiting 2 seconds...")
                    time.sleep(2)
                    
    def _task_thread(self, data, task_callable, task_index): 
        """Thread function which processes a single task."""
        result = task_callable(data)
        self._store_result(result, task_index)
        self._n_active_threads -= 1
