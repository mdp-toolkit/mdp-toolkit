"""
Adapters for the Parallel Python library (http://www.parallelpython.com).

This python file also serves as the emergency kill script for remote slaves.
If it is run the kill_slaves function is called.
"""

import sys
import os
import inspect
import logging

if __name__ == "__main__":
    module_file = os.path.abspath(inspect.getfile(sys._getframe(0)))
    module_path = os.path.dirname(module_file)
    sys.path.append(module_path.split("mdp")[0])

import time
import subprocess
import signal
import traceback
import itertools

import scheduling
import pp

# TODO: modify pythonpath when starting workers, then specify module mdp...
#    write a python wrapper for starting the worker which modifies the sys.path

# TODO: list of computers as dict with int for number of processes?

class PPScheduler(scheduling.Scheduler):
    """Adaptor scheduler for the parallel python scheduler.
    
    This scheduler is a simple wrapper for a pp server. A pp server instance
    has to be provided.
    """
    
    def __init__(self, ppserver, max_queue_length=1,
                 result_container=scheduling.ListResultContainer(), 
                 copy_callable=True,
                 verbose=False):
        """Initialize the scheduler.
        
        ppserver -- Parallel Python Server instance.
        max_queue_length -- How long the queue can get before add_task blocks.
        result_container -- ResultContainer used to store the results.
        copy_callable -- If True then the callable will be copied before being 
            called (default value is True).
        verbose -- If True to get progress reports from the scheduler.
        """
        super(PPScheduler, self).__init__(result_container=result_container,
                                          copy_callable=copy_callable,
                                          verbose=verbose)
        self.ppserver = ppserver
        self.max_queue_length = max_queue_length
    
    def _process_task(self, data, task_callable, task_index):
        """Non-blocking processing of tasks.
        
        Depending on the scheduler state this function is non-blocking or
        blocking. One reason for blocking can be a full task-queue.
        """
        if self.copy_callable:
            task_callable = task_callable.copy()
        task = (data, task_callable, task_index)
        def execute_task(task):
            """Call the first args entry and return the return value."""
            data, task_callable, task_index = task
            return task_callable(data), task_index
        task_submitted = False
        while not task_submitted:
            if len(self.ppserver._Server__queue) > self.max_queue_length:
                # release lock for other threads and wait
                self.lock.release()
                time.sleep(0.5)
                self.lock.acquire()
            else:
                # release lock to enable result storage
                self.lock.release()
                # the inner tuple is a trick to prevent introspection by pp
                # this forces pp to simply pickle the object
                self.ppserver.submit(execute_task, args=(task,),
                                     callback=self._pp_result_callback)
                task_submitted = True
                
    def _pp_result_callback(self, result):
        """Calback method for pp to unpack the result and the task id.
        
        This method then calls the normal _store_result method.
        """
        self._store_result(result[0], result[1])
        
    def shutdown(self):
        """Call destroy on the ppserver."""
        self.ppserver.destroy()
    
    
class LocalPPScheduler(PPScheduler):
    """Usees a local pp server to distribute the work across cpu cores.
    
    The pp server is created automatically instead of beeing provided by the
    user (in contrast to PPScheduler).
    """
    
    def __init__(self, ncpus="autodetect", max_queue_length=1,
                 result_container=scheduling.ListResultContainer(),
                 copy_callable=True,
                 verbose=False):
        """Create an internal pp server and initialize the scheduler.
        
        ncpus -- Integer or 'autodetect', specifies the number of processes
            used.
        max_queue_length -- How long the queue can get before add_task blocks.
        result_container -- ResultContainer used to store the results.
        copy_callable -- If True then the callable will be copied before being 
            called (default value is True).
        verbose -- If True to get progress reports from the scheduler.
        """
        ppserver = pp.Server(ncpus=ncpus,
                             loglevel=logging.INFO,
                             logstream=sys.stdout)
        super(LocalPPScheduler, self).__init__(ppserver=ppserver,
                                          max_queue_length=max_queue_length,
                                          result_container=result_container,
                                          copy_callable=copy_callable,
                                          verbose=verbose)
        
    
# filename used to store the slave info needed for a complete kill
SLAVES_TEMPFILE = "networkslaves.txt"
# default secret
SECRET = "rosebud"
    
class NetworkPPScheduler(PPScheduler):
    """Scheduler which can manage pp remote servers.
    
    The remote slave servers are automatically started and killed at the end.
    """
    
    def __init__(self, max_queue_length=1,
                 result_container=scheduling.ListResultContainer(),
                 copy_callable=True,
                 verbose=False,
                 remote_slaves=None,
                 port=50017,
                 secret=SECRET,
                 nice=-19,
                 timeout=3600,
                 temp_filename=None,
                 n_local_workers=0,
                 source_paths=None,
                 remote_python_executable=None):
        """Initialize the remote slaves and create the internal pp scheduler.
        
        result_container -- ResultContainer used to store the results.
        copy_callable -- If True then the callable will be copied before being 
            called (default value is True).
        verbose -- If True to get progress reports from the scheduler.
        remote_slaves -- List of tuples, the first tuple entry is a string
            containing the name or IP adress of the slave, the second entry
            contains the number of processes (i.e. the pp ncpus parameter).
            The second entry can be None to use 'autodetect'.
        n_local_workers -- Value of ncpus for this machine.
        secret -- Secret password to secure the remote slaves.
        source_paths -- List of paths that will be appended to sys.path in the
        slaves.
        """
        self._remote_slaves = remote_slaves
        self._running_remote_slaves = None  # list of strings 'address:port'
        # list with processes for the ssh connections to the slaves
        self._ssh_procs = None
        self._remote_pids = None  # list of the pids of the remote servers
        self._port = port
        if temp_filename is not None:
            self.temp_filename = temp_filename
        else:
            # store the tmp file in this dir so that the kill script works
            module_file = os.path.abspath(inspect.getfile(sys._getframe(0)))
            module_path = os.path.dirname(module_file) 
            self.temp_filename = os.path.join(module_path, SLAVES_TEMPFILE)
        self._secret = secret
        self._slave_nice = nice
        self._timeout = timeout
        self._source_paths = source_paths
        self._python_executable = remote_python_executable
        module_file = os.path.abspath(inspect.getfile(sys._getframe(0)))
        self._script_path = os.path.dirname(module_file)
        # start ppserver
        self._start_slaves()
        ppslaves = tuple(["%s:%d" % (address, self._port) 
                          for address in self._running_remote_slaves])
        ppserver = pp.Server(ppservers=ppslaves,
                             ncpus=n_local_workers,
                             secret=self._secret,
                             loglevel=logging.INFO,
                             logstream=sys.stdout)
        super(NetworkPPScheduler, self).__init__(ppserver=ppserver,
                                          max_queue_length=max_queue_length,
                                          result_container=result_container,
                                          copy_callable=copy_callable,
                                          verbose=verbose)
    
    def shutdown(self):
        """Shutdown all slaves."""
        
        for ssh_proc, pid, address in itertools.izip(self._ssh_procs, 
                                self._remote_pids, self._running_remote_slaves):
            print "killing slave " + address 
            ssh_proc.stdin.write("kill %d\n" % pid)
            ssh_proc.stdin.flush()
            # a SIGKILL might prevent the kill command transmission
            os.kill(ssh_proc.pid, signal.SIGQUIT)
        print "all slaves killed"
        super(NetworkPPScheduler, self).shutdown()
        
    def _start_slaves(self):
        """Start remote slaves.
        
        The slaves that could be started are stored in a textfile, in the form
        name:port:pid
        """
        tempfile = open(self.temp_filename, "w")
        try:
            self._running_remote_slaves = []
            self._remote_pids = []
            self._ssh_procs = []
            for (address, ncpus) in self._remote_slaves:
                ssh_proc, pid = start_slave(
                                    address=address, port=self._port, 
                                    ncpus=ncpus, 
                                    secret=self._secret, 
                                    nice=self._slave_nice,
                                    script_path=self._script_path,
                                    source_paths=self._source_paths,
                                    python_executable=self._python_executable, 
                                    timeout=self._timeout)
                if pid is not None:
                    tempfile.write("%s:%d:%d\n" % (address, pid, ssh_proc.pid))
                self._running_remote_slaves.append(address)
                self._remote_pids.append(pid)
                self._ssh_procs.append(ssh_proc)
        finally:
            tempfile.close()
        

### Helper functions ###
            
def start_slave(address, port, ncpus="autodetect", secret=SECRET, timeout=3600, 
                nice=-19, script_path="", 
                source_paths=None, python_executable=None):
    """Start a single remote slave.
    
    The return value is a tuple of the ssh process handle and the remote pid.
    
    script_path -- Path to pp slave script file (pp_slave_script).
    source_paths -- List of paths that will be appended to sys.path in the
        slaves.
    """
    try:
        if python_executable is None:
            python_executable = sys.executable
        print "starting " + address + " ..."
        proc = subprocess.Popen(["ssh","-T", "%s" % address],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        proc.stdin.write("cd %s\n" % script_path)
        cmd = (python_executable + " pp_slave_script.py  %d %d %d %s %d" % 
               (nice, port, timeout, secret, ncpus))
        proc.stdin.write(cmd + "\n")
        # send sys_paths
        source_paths = [python_executable,] + source_paths
        if source_paths is not None:
            for sys_path in source_paths:
                proc.stdin.write(sys_path + "\n")
        proc.stdin.write("_done_" + "\n")
        # print status message from slave
        sys.stdout.write(address + ": " + proc.stdout.readline())
        # get PID for remote slave process
        pid = int(proc.stdout.readline())
        return (proc, pid)
    except:
        print "Initialization of slave %s has failed." % address
        traceback.print_exc()
        return None
    
def kill_slaves(temp_filename=SLAVES_TEMPFILE):
    """Kill all remote slaves which are stored in the tempfile."""
    tempfile = open(temp_filename)
    try:
        for line in tempfile:
            address, pid, ssh_pid = line.split(":")
            pid = int(pid)
            ssh_pid = int(ssh_pid)
            # open ssh connection to to kill remote slave
            proc = subprocess.Popen(["ssh","-T", address],
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            proc.stdin.write("kill %d\n" % pid)
            proc.stdin.flush()
            # kill old ssh connection
            try:
                os.kill(ssh_pid, signal.SIGKILL)
            except:
                pass
            # a kill might prevent the kill command transmission
            # os.kill(proc.pid, signal.SIGQUIT)
            print "killed slave " + address + " (pid %d)" % pid
        print "all slaves killed."
    finally:
        tempfile.close()
        

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        kill_slaves()

