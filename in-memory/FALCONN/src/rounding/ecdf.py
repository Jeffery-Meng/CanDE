import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
from sys import argv


data = np.loadtxt(argv[1])
prefix = argv[1].split(".")[0]
near = data[0]
far = data[1]
print(near.shape, far.shape)

def plot_ecdf(data):
    ecdf = sm.distributions.ECDF(data)
    x = np.linspace(min(data), max(data), 1000)
    y = ecdf(x)
    plt.step(x, y)
    
plot_ecdf(near)
plot_ecdf(far)
plt.savefig(prefix+".png")

ecdf = sm.distributions.ECDF(far)
for perc in np.arange(10, 100, 10):
    x = np.percentile(near, perc)
    y = ecdf(x)
    print("{:8.4f}\t{:8.4f}\t{:8.4f}".format(perc, x, y))
