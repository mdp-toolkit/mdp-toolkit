"""
Framework for order preserving execution of jobs by a scheduler.

This is achieved by labeling the data chunks with marker numbers,
i.e. data = (marker, x)
All the necessary classes are provided by this module.
"""

import scheduling


class OrderedListResultContainer(scheduling.ListResultContainer):
    """Result container which helps to preserve the order of the data blocks.
    
    The results are labeled and finally sorted before being returned.
    """
    
    def get_results(self):
        """Return the ordered results."""
        marked_results = self._results
        self._results = []
        def compare_marker(x, y):
            return x[0] - y[0]
        marked_results.sort(compare_marker)
        return zip(*marked_results)[1]
    
    
class OrderedJob(scheduling.Job):
    """Abstract base class to remove the order number from the data
    and then reapply it.
    
    OrderedIterable can be used to create the marked data for this Job.
    """
    
    def __init__(self, marked_x):
        """Split up marker and data and store them."""
        self._marker = marked_x[0]
        self._x = marked_x[1]
        
    def _apply_marker(self, result):
        """Rejoin the maker with the corresponding result."""
        return (self._marker, result)
    
    def __call__(self):
        """Overwrite this method to implement the actual calculation.
        
        Before returning the result apply_marker must be used.
        """
        return self._apply_marker(result=self._x)
        

class OrderedIterable(object):
    """Adaptor class to add an order marker to any iterable."""
    
    def __init__(self, data_iter):
        """Store the iterable which will be wrapped by this class."""
        self.data_iter = data_iter
        
    def __iter__(self):
        """Yield the marked data chunks from the data_iter."""
        for i_data, data in enumerate(self.data_iter):
            yield (i_data, data)
    