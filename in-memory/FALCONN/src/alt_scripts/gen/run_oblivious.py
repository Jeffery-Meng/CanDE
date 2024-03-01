def run_obliv():
    for qid in range(1000):
        yield ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/distance_hist_oblivious.py",
            str(qid)]

def run_obliv2():
    for qid in range(1000):
        yield ["python3", "/media/mydrive/distribution/ann-codes/in-memory/FALCONN/src/kde_scripts/distance_hist_bayesian2.py",
            str(qid)]