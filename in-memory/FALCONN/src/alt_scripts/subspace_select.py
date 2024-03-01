import os.path
import sys

path = "/media/mydrive/ann-codes/in-memory/EXPERIMENTS/subspace_tuning"
dataset = sys.argv[1]
r = sys.argv[2]
target_recalls = [0.4, 0.5, 0.6, 0.7, 0.8]
lowest_ratios = [10] * 5
final_para = [0]*3

prefix = dataset + "-" + r
for file in os.listdir(path):
    if prefix not in file:
        continue
    with open(os.path.join(path, file), "r") as fin:
        fin.readline()
        line = fin.readline().split()
        try:
            l = float(line[0])
            m = float(line[1])
            w = float(line[2])
            recall = float(line[3])
            ratio = float(line[4])
        except:
            continue
    
    for i in range(len(target_recalls)):
        if recall <= target_recalls[i] - 0.001:
            break
        if ratio < lowest_ratios[i]:
            lowest_ratios[i] = ratio
            final_para = [l,m,w]
            

print(lowest_ratios)
print("parameter (l,m,W) is:")
print(final_para)