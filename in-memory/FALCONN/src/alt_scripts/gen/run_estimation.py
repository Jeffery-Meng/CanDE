def run_estimators(dataset):
    """   for i in range(1000):
        yield ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/distance_hist.py",
            "EXPERIMENTS/{}/ann_ps2.json".format(dataset), "-q", str(i)]
    for i in range(1000):
        yield ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/distance_hist_triangle.py",
            dataset, str(i)] """
    for i in range(1000):
        yield ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/distance_hist_bayesian2.py",
            dataset, str(i)]

def run_gamma(dataset):
    for i in range(1000):
        yield ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/gamma.py",
            dataset, str(i)]

def run_bayesian(dataset):
    for i in range(1000):
        yield ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/distance_hist_bayesian3.py",
            dataset, str(i)]

import subprocess, os

if __name__ == "__main__":
    exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
    for dataset_name in ["deep", "gist", "random"]:
        exp_path = os.path.join(exp_path, dataset_name)
        auto_param_cmd = ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/command_console.py"]
        auto_param_line = __file__ + " " + "run_bayesian" + " "
        auto_param_line += dataset_name
        auto_param_line += "\n"
        subprocess.run(auto_param_cmd, input=auto_param_line.encode())