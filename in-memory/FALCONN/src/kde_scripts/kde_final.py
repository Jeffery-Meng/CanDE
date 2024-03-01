import os

exp_path = "/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS"
datasets = ["audioN", "enronN", "deepN", "gistN", "gloveN", "siftN", "mnistN", "treviN"]

results = []

for dataset in datasets:
    cur_path = os.path.join(exp_path, dataset)
    results.append([dataset])
    with open(os.path.join(cur_path, "kde_mle.txt")) as fin:
        results[-1].extend([float(x) for x in fin.readline().split()[:2]])
    cur_path = os.path.join(cur_path, "summary")
    tbl_files = os.listdir(cur_path)
    best_mre = 1000
    # for fl in tbl_files:
    #     if "kde_v3_mres" in fl:
    #         with open(os.path.join(cur_path, fl)) as fin:
    #             mre = float(fin.readline().split()[1])
    #         if mre < best_mre:
    #             tbl_num = int(fl.split("_")[-1][:-4])
    #             best_mre = mre
    with open(os.path.join(cur_path, "kde_v3_mres_100.txt")) as fin:
         mre = float(fin.readline().split()[1])
    results[-1].append(mre)
    cur_path = os.path.join(cur_path, "..")

    with open(os.path.join(cur_path, "kde_sample.txt")) as fin:
         mre = [float(x) for x in fin.readline().split()]
    results[-1].extend([mre])
    with open(os.path.join(cur_path, "kde_sample_tbl.txt")) as fin:
        mre = [float(x) for x in fin.readline().split()]
    results[-1].extend([mre])

with open(os.path.join(exp_path, "kde_results3.txt"), "w") as fout:
    for r in results:
        fout.write(str(r) + "\n")