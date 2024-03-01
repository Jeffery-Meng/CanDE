import os, errno, time, subprocess, pathlib

def is_running(pid):        
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
    return True

while True:
    if is_running(29601):
        time.sleep(10)
    else:
        break

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent.parent / "build")
subprocess.run(["make", "-C", bin_path])
subprocess.run(["python3", str(this_path / "run_kde_tbl3.py")])