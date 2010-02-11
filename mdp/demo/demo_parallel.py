"""
Demo for the speedup that the parallelization can offer.

The parallelization usually pays off when there is lots of data (so that
stop_training is fast compared to the train calls) and when the processing
of the data chunks is costly enough to dominate over the cost of sending the
training data to the parallel worker processes (e.g. if a non-linear exapansion
is involved then this increases the training time without increasing the
data volume).

With the current example values the parallelized version is faster by a factor
of 2.2 on an Intel Core 2 Quad @ 2.66 GHz. The parallel training time is
33 seconds. While this factor is much smaller than the theoretical maximum of 4
it can be increased by using more data chunks and a larger input dimension.
For training times over 10 minutes one can thus achieve a speedup factor > 3.

Parallelization only pays off for large training tasks, which is actually not
that bad since this is where the speedup is most needed.
"""

import numpy as np
import mdp
import time

n_processes = 2

n_chunks = 10
n_chunk_samples = 5000
n_dim = 30

## threads

print "starting thread parallel training..."
flow = mdp.parallel.ParallelFlow([mdp.nodes.SFA2Node()])
x_iter = [np.random.random((n_chunk_samples, n_dim)) for _ in range(n_chunks)]
scheduler = mdp.parallel.ThreadScheduler(n_threads=n_processes, verbose=True)

start_time = time.time()
flow.train([x_iter], scheduler)
parallel_time = time.time() - start_time
scheduler.shutdown()
print "thread parallel in %.3f secs" % parallel_time

## processes

print "starting process parallel training..."
flow = mdp.parallel.ParallelFlow([mdp.nodes.SFA2Node()])
x_iter = [np.random.random((n_chunk_samples, n_dim)) for _ in range(n_chunks)]
scheduler = mdp.parallel.ProcessScheduler(n_processes=n_processes, verbose=True)

start_time = time.time()
flow.train([x_iter], scheduler)
parallel_time = time.time() - start_time
scheduler.shutdown()
print "process parallel in %.3f secs" % parallel_time

## sequential training

print "starting sequential training..."
flow = mdp.Flow([mdp.nodes.SFA2Node()])

start_time = time.time()
flow.train([x_iter])
nonparallel_time = time.time() - start_time
print "sequential in %.3f secs" % nonparallel_time

speedup = 1.0 * nonparallel_time / parallel_time
print "speedup factor: %.1f (%d processes)" % (speedup, n_processes)
