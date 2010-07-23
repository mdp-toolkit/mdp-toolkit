import mdp.parallel as parallel
from _tools import *

from test_pp_local import requires_parallel_python

remote_slaves = [("sherrington", 1),
                 ("weismann", 2)]

python_executable = "/home/wilbert/bin/python"
sys_paths = ["/home/wilbert/develop/workspace/MDP",
             "/home/wilbert/develop/workspace/parallelpython/src/pp"]

@requires_parallel_python
def test_simple(self):
    scheduler = pp_support.NetworkPPScheduler(
        remote_slaves=remote_slaves,
        timeout=60,
        source_paths=sys_paths,
        remote_python_executable=python_executable,
        verbose=False)
    # process jobs
    for i in range(30):
        scheduler.add_task(i, parallel.SqrTestCallable())
    results = scheduler.get_results()
    scheduler.shutdown()
    # check result
    results.sort()
    results = n.array(results)
    assert n.all(results[:6] == n.array([0,1,4,9,16,25]))
