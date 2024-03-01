def map_calc(candidates : list, ground_truth : list, K : int) -> float:
    correct_num = 0
    sum_precision = 0.
    for i in range(K):
        if candidates[i] != ground_truth[i]:
            continue
        else:
            correct_num += 1
            recall = correct_num / i
            sum_precision += recall
    return sum_precision / K