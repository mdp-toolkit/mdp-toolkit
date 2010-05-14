"""
Demo for the speedup that the parallelization can offer.

This demo compares the thread and the process based paralelisation with the
non-parallel version.

The process based-parallelization only pays off when there is lots of data (so
that stop_training is fast compared to the train calls) and when the processing
of the data chunks is costly enough to dominate over the cost of sending the
training data to the parallel worker processes (e.g. if a non-linear exapansion
is involved then this increases the training time without increasing the data
volume).

In principle the process-based parallelisation can beat the thread-based one if
the GIL prevents the threads from using the cores effectively.

The thread-based parallelisation carries some overhead over the non-parallel
version because it has to fork the nodes and creates threads.
"""

import numpy as np
import mdp
import time

n_threads = 2
n_processes = 2

n_chunks = 20
n_chunk_samples = 1000
n_dim = 30

nonparallel_time = None
thread_time = None
process_time = None

## threads

print "starting thread parallel training..."
flow = mdp.parallel.ParallelFlow([mdp.nodes.SFA2Node()])
x_iter = [np.random.random((n_chunk_samples, n_dim)) for _ in range(n_chunks)]
scheduler = mdp.parallel.ThreadScheduler(n_threads=n_threads, verbose=True)

start_time = time.time()
flow.train([x_iter], scheduler)
thread_time = time.time() - start_time
scheduler.shutdown()
print "thread parallel in %.3f secs" % thread_time

## processes

print "starting process parallel training..."
flow = mdp.parallel.ParallelFlow([mdp.nodes.SFA2Node()])
x_iter = [np.random.random((n_chunk_samples, n_dim)) for _ in range(n_chunks)]
scheduler = mdp.parallel.ProcessScheduler(n_processes=n_processes, verbose=True)

start_time = time.time()
flow.train([x_iter], scheduler)
process_time = time.time() - start_time
scheduler.shutdown()
print "process parallel in %.3f secs" % process_time

## sequential training

print "starting sequential training..."
flow = mdp.Flow([mdp.nodes.SFA2Node()])

start_time = time.time()
flow.train([x_iter])
nonparallel_time = time.time() - start_time
print "sequential in %.3f secs" % nonparallel_time

if nonparallel_time and thread_time:
    speedup = 1.0 * nonparallel_time / thread_time
    print "thread speedup factor: %.1f (%d threads)" % (speedup, n_threads)
if nonparallel_time and process_time:
    speedup = 1.0 * nonparallel_time / process_time
    print "process speedup factor: %.1f (%d processes)" % (speedup, n_processes)
