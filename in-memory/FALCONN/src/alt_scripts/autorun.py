import subprocess
import os, json, sys
import time
import numpy as np

dataset = sys.argv[1]

dim_as = [2,4,8,16,32,64]

max_processes = 1
processes = set()

start_time = time.time()
cnt = 0

for dima in dim_as:
   
    processes.add(subprocess.Popen(["python3", "linear_scan.py", dataset, str(dima)]))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([
            p for p in processes if p.poll() is not None])
    cnt += 1
    
print("Finished {} subprocesses in {} seconds.".format(cnt, time.time()-start_time))