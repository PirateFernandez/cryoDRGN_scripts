import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
###simple way to open pkl files coming from cryoDRG
###
with open('/Users/israel_CUMC/Documents/XXX/cryoDrg/00_vae128_z1/z.pkl', 'rb') as f:
	X = pickle.load(f)
with open('/Users/israel_CUMC/Documents/XXX/cryoDrg/00_vae128_z8/z.pkl', 'rb') as f:
	Y = pickle.load(f)
### z.pkl are np arrays in this case of 8 rows (dimensions) and ~130K inputs per row which are particles
### with this list comprehension re-order the np array in a 1D list. I guess I could do it also with np.reshape
list_data = [Y[:,i] for i in range(0,8)]
###I was thinking on a way to directly map each entry of the list to a specific subplot. Zipping is the solution as it creates a tuple of ax, data, ind
### I can iterate through this triple tupple
fig, axes = plt.subplots(2,4)
fig.suptitle("Histogram")
for ax, data, ind in zip(axes.ravel(), list_data, range(0,8)):
	ax.hist(data, 50)
	if ind!=0 and ind!=4:
		ax.set_yticks([])
	ax.set_ylim(-1,15000)
	ax.set_xlim(-5,5)
plt.show()
