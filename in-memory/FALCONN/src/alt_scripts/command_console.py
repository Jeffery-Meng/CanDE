""" An automatic console for running batch jobs in parallel. 
You can add new jobs when other jobs are running. """

import subprocess
import os, json, sys
import time
import numpy as np
import importlib.util
import select
from collections import deque

processes = set()
max_processes = 30

if "--maxprocess" in sys.argv:
    idx = sys.argv.index("--maxprocess")
    max_processes = int(sys.argv[idx+1])

start_time = time.time()
cnt = 0

def load_generator(path, generator_name):
    module_name = "temp_module"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = getattr(module, generator_name)
    return (generator(), generator_name)

class JobGenerator:
    def __init__(self):
        self.generator_queue = deque()

    def __next__(self):
        while self.generator_queue:
            try:
                next_job = next(self.generator_queue[0][0])
                return next_job
            except StopIteration:
                self.generator_queue.popleft()
        raise StopIteration

    def append(self, generator_with_name):
        self.generator_queue.append(generator_with_name)

    def __len__(self):
        return len(self.generator_queue)
    
    def remove(self, name):
        for job in self.generator_queue:
            if job[1] == name:
                self.generator_queue.remove(job)
                return

job_queue = JobGenerator()

def handle_command(line):
    command = line.split[1]
    if command == "abort":
        name = line.split[2]
        job_queue.remove(name)
        print("{} removed from job queue. Note: active jobs will not be canceled.".format(name))
        print("Please use `pkill -f [pattern]` command for that.")
        return
    if command == "set":
        attribute = line.split[2]
        if attribute == "max_processes":
            global max_processes
            max_processes = int(line.split[3])
            return
    print("Unknown Command!")

finished = False
while not finished:
    try:
        initial_job = input("Please add your job.\n Format: [path to .py] [generator name]\n")
        job_queue.append(load_generator(*initial_job.split()))
        finished = True
    except:
        continue

while True:
    # update job status
    processes.difference_update([
                    p for p in processes if p.poll() is not None])
    # start job execution
    while len(processes) < max_processes:
        try:
            processes.add(subprocess.Popen(next(job_queue)))
            cnt += 1
        except StopIteration:  # no more jobs to add
            break

    # wait for 10s for user input
    try:
        new_job, _, _ = select.select([sys.stdin], [], [], 10)
        if new_job:
            line = sys.stdin.readline()
            if (line[0] == "*"):
                handle_command(line)
            else:
                job_queue.append(load_generator(*line.split()))
        sys.stdin.flush()
    except Exception as e:
        print(e)
        continue

    # exit program if both job queue and generator queue are empty
    if not job_queue and not processes:
        break

print("No active jobs, quiting the console.")
print("Finished {} subprocesses in {} seconds.".format(cnt, time.time()-start_time))