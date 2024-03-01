import subprocess
import pathlib
import os, sys
import json
import numpy as np
import fvecs

from all_one import all_one

# Run ground truth for KDE.

exp_path = sys.argv[1]
exp_path = str(pathlib.Path(exp_path).resolve())

this_path = pathlib.Path(__file__).parent.resolve()
bin_path = str(this_path.parent.parent / "build")

conf_path = os.path.join(exp_path, "config", "kde_timetest_{}.json".format(
    131072))
subprocess.run([os.path.join(bin_path, "falconn_disthist_tbl2"), conf_path])
config_file = os.path.join(exp_path,  "config/kde_tbl8.json") 

subprocess.run([os.path.join(bin_path, "falconn_kde_v3"), config_file])
