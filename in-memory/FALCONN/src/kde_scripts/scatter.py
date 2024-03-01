N=600

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

x = rng.normal(size=N)

y =  rng.normal(size=N)
plt.scatter(x, y)
ax = plt.gca()
plt.savefig("/media/mydrive/distribution/ann-codes/in-memory/EXPERIMENTS/plot2/plt3.pdf")
plt.show()


