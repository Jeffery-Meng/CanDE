import sys

cnt = 0
time = 0.0
tv = 0.0
num = 0
with open(sys.argv[1]) as fin:
    for line in fin.readlines():
        if cnt % 2 == 1:
            cnt += 1
            continue
        ls = line.split("\t")
        failure_cnt = int(ls[5])
        if failure_cnt > 5000:
            cnt += 1
            continue
        time += float(ls[1])
        tv += float(ls[3])
        num += 1
        cnt += 1
time /= num * 1e5
tv /= num

print(num, time, tv)