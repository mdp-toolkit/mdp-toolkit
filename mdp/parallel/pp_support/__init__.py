"""
This package provides experimentel support for the 'Parallel Python' libray.

The 'Parallel Python' open source libray is available at www.parallelpython.com.
When it is installed you can try to use it with the adapter schedulers provided
here.

Warning: This package is still in an experimentel/alpha state, so use with care!
But if you are very interested in using 'Parallel Python' it should already be
quite useful. It should basically work and has passed limited testing.
"""

from pp_schedule import (PPScheduler, LocalPPScheduler, NetworkPPScheduler,
                         start_slave, kill_slaves)

del pp_schedule