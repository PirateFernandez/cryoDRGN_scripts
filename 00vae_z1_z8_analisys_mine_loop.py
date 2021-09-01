import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
with open('/Users/israel_CUMC/Documents/40S-crpv/cryoDrg/00_vae128_z1/z.pkl', 'rb') as f:
	X = pickle.load(f)
with open('/Users/israel_CUMC/Documents/40S-crpv/cryoDrg/00_vae128_z8/z.pkl', 'rb') as f:
	Y = pickle.load(f)
list_data = [Y[:,i] for i in range(0,8)]
fig, axes = plt.subplots(2,4)
fig.suptitle("Histogram")
for ax, data, ind in zip(axes.ravel(), list_data, range(0,8)):
	ax.hist(data, 50)
	if ind!=0 and ind!=4:
		ax.set_yticks([])
	ax.set_ylim(-1,15000)
	ax.set_xlim(-5,5)
plt.show()
