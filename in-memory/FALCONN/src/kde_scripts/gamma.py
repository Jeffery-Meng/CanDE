import numpy as np
import sys, json
from scipy.stats import gamma
from fvecs import *


# num_samples: Number of samples for gamma distribution, 1% of data points should be enough
# Generate mre for each query
def gamma_dists(input_bins, query_dists, num_samples=None):
    re_all = np.zeros(len(input_bins)-1)

    if num_samples is None:
        num_samples = int(len(query_dists) * 0.01)

    sample = np.random.choice(np.arange(len(query_dists)),num_samples,replace=False)
    fit_a, fit_loc, fit_scale=gamma.fit(query_dists[sample])

    y = np.zeros(len(input_bins)-1)
    for j in range(len(input_bins)-1):
        y[j] = gamma.cdf(input_bins[j+1],fit_a,loc=fit_loc,scale=fit_scale)-gamma.cdf(input_bins[j],fit_a,loc=fit_loc,scale=fit_scale)
        y[j] *= len(query_dists)

    gnd=  np.histogram(query_dists, bins=input_bins)[0]
    diff = np.abs(y - gnd, dtype=np.float64)
    re_all = np.divide(diff, gnd, out=np.zeros_like(diff), where=gnd!=0)
    
    return re_all

if __name__ == "__main__":

    exp_path = sys.argv[1]
    conf_path = exp_path + "/ann_gamma.json"

    with open(conf_path) as fin:
        conf = json.load(fin)
    distance = fvecs_read(exp_path + "/distances.fvecs")
    input_bins = np.arange(*conf["histogram bins"])

    mre = None
    for qid in range(conf["testing size"]):
        print(qid)
        mre_na = gamma_dists(input_bins, distance[qid, :])
        mre = mre_na if mre is None else mre + mre_na
    mre /= conf["testing size"]
    
    bin_xs = [(input_bins[i] + input_bins[i+1]) / 2 for i in range(len(input_bins)-1)]
    bin_xs = np.array(bin_xs)
    with open(conf["result filename"], "w") as fout:
        np.savetxt(fout, bin_xs[None])
    with open(conf["result filename"], "a") as fout:
        np.savetxt(fout, mre[None])
