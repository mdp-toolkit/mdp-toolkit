"""
This module contains the basic classes for task processing via a scheduler.
"""

import thread
import time
import copy

# TODO: remove copy_callable flag, instead override copy() method in
#    TaskCallable


class ResultContainer(object):
    """Abstract base class for result containers."""
    
    def add_result(self, result_data, task_index):
        """Store a result in the container."""
        pass
    
    def get_results(self):
        """Return results and reset container."""
        pass
    

class ListResultContainer(ResultContainer):
    """Basic result container using simply a list."""
    
    def __init__(self):
        super(ListResultContainer, self).__init__()
        self._results = []
        
    def add_result(self, result, task_index):
        """Store a result in the container."""
        self._results.append(result)
        
    def get_results(self):
        """Return the list of results and reset this container.
        
        Note that the results are stored in the order that they come in, which
        can be different from the orginal task order.
        """
        results = self._results
        self._results = []
        return results
    
    
class OrderedResultContainer(ListResultContainer):
    """Default result container with automatic restoring of the result order.
    
    In general the order of the incoming results in the scheduler can be
    different from the order of the tasks, since some tasks may finish quicker
    than other tasks. This result container restores the original order.
    """
    
    def __init__(self):
        super(OrderedResultContainer, self).__init__()
        
    def add_result(self, result, task_index):
        """Store a result in the container.
        
        The task index is also stored and later used to reconstruct the
        original task order.
        """
        self._results.append((result, task_index))
        
    def get_results(self):
        """Sort the results into the original order and return them in list."""
        results = self._results
        self._results = []
        def compare_marker(x, y):
            return x[1] - y[1]
        results.sort(compare_marker)
        return zip(*results)[0]
    
    
class TaskCallable(object):
    """Abstract base class for callables."""
    
    def __call__(self, data):
        return data
    
    def copy(self):
        """Create a copy of this callable.
        
        This is required if copy_callable is set to True in the scheduler (e.g. 
        if caching is used during training of a flow).
        """
        return copy.deepcopy(self)
    
    
class SqrTestCallable(TaskCallable):
    """Test callable to be used where a function cannot be used.
    
    This is for example the case in schedulers which pickle the callable.
    """
    
    def __call__(self, data):
        """Return the squared data."""
        return data**2
    

class Scheduler(object):
    """Base class and trivial implementation for schedulers.
    
    New tasks are added with add_task(data, callable).
    get_results then returns the results (and locks if tasks are
    pending).
    
    In this simple scheduler implementation the tasks are simply executed in the 
    add_task method.
    """

    def __init__(self, result_container=None, copy_callable=True,
                 verbose=False):
        """Initialize the scheduler.
        
        result_container -- Instance of ResultContainer that is used to store
            the results (default is None, in which case a ListResultContainer
            is used).
        copy_callable -- If True then the callable will be copied before being 
            called (default value is True). Setting this to False might be more
            efficient cases were this is not needed. The callable must have a 
            working copy method when copy_callable is True.
            Note that derived scheduler classes might not support this 
            feature (e.g. because they choose to always copy the callable). In
            this case they should set the copy_callable flag to True.
        verbose -- If True then status messages will be printed to sys.stdout.
        """
        if result_container is None:
            result_container = OrderedResultContainer()
        self.result_container = result_container
        self.copy_callable = copy_callable
        self.verbose = verbose
        self.n_open_tasks = 0  # number of tasks that are currently running
        # count the number of submitted tasks, 
        # this value is also used as task index
        self.task_counter = 0  
        self.lock = thread.allocate()  # general lock for this class
        self._last_callable = None  # last callable is stored
        # task index of the _last_callable, can be *.5 if updated between tasks
        self._last_callable_index = -1.0
           
    def add_task(self, data, task_callable=None):
        """Add a task to be executed.
        
        data -- Data for the task.
        task_callable -- A callable, which is called with the data. If it is 
            None (default value) then the last provided callable is used.
        
        The callable together with the data constitutes the task.
        In this simple implementation the task is simply executed. The method
        blocks if another task is already running (which may happen if multiple 
        threads are used). 
        
        This method is overridden in more complex schedulers. Generally this
        method is potentially blocking (e.g. when the task queue is full).
        """
        self.lock.acquire()
        if task_callable is None:
            if self._last_callable is None:
                raise Exception("No task_callable specified and " + 
                                "no previous callable available.")
        self.n_open_tasks += 1
        self.task_counter += 1
        task_index = self.task_counter
        if task_callable is None:
            # use the _last_callable_index in _process_task to
            # decide if a cached callable can be used 
            task_callable = self._last_callable
        else: 
            self._last_callable = task_callable
            self._last_callable_index = self.task_counter
        self._process_task(data, task_callable, task_index)
        
    def _process_task(self, data, task_callable, task_index):
        """Process the task and store the result.
        
        Warning: When this method is entered is has the lock, the lock must be
        released here.
        
        You can override this method for custom schedulers.
        """
        if self.copy_callable:
            task_callable = task_callable.copy()
        result = task_callable(data)
        self.lock.release()
        self._store_result(result, task_index)
        
    def set_task_callable(self, task_callable, copy_callable=None):
        """Set the callable that will be used if no task_callable is given.
        
        task_callable -- Callable that will be used as default unless a new
            task_callable is given.
        copy_callable -- New value for the copy switch, if None (default value)
            the value is not changed.
        """
        self.lock.acquire()
        self._last_callable = task_callable
        # set _last_callable_index to half value since the callable is newer 
        # than the last task, but not newer than the next incoming task
        self._last_callable_index = self.task_counter + 0.5
        if copy_callable is not None:
            self.copy_callable = copy_callable
        self.lock.release()
        
    def _store_result(self, result, task_index):
        """Store a result in the internal result container.
        
        result -- Tuple of result data and task index.
        
        This function blocks to avoid any problems during result storage.
        """
        self.lock.acquire()
        self.result_container.add_result(result, task_index)
        if self.verbose:
            print "    finished task no. %d" % task_index
        self.n_open_tasks -= 1
        self.lock.release()
        
    def get_results(self):
        """Get the accumulated results from the result container.
        
        This function blocks if there are open tasks. 
        """
        while True:
            self.lock.acquire()
            if self.n_open_tasks == 0:
                results = self.result_container.get_results()
                self.lock.release()
                return results
            else:
                self.lock.release();
                time.sleep(1); 

    def shutdown(self):
        """Controlled shutdown of the scheduler.
        
        This method should always be called when the scheduler is no longer 
        needed and before the program shuts down! Otherwise one might get
        error messages.
        """
        pass
