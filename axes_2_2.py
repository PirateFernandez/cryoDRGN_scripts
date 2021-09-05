import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model
with open('/Users/israel_CUMC/Documents/###/cryoDrg/00_vae128_z1/z.pkl', 'rb') as f:
	X = pickle.load(f)
with open('/Users/israel_CUMC/Documents/###/cryoDrg/00_vae128_z8/z.pkl', 'rb') as f:
	Y = pickle.load(f)
####create a list from the np array of cryodrg pkl file using list comprehension where each dimension is one list entry####
list_data = [Y[:,i] for i in range(0,8)]
####in order to compare dimension 1 and 2, create a np array for use with scipi.stats.linregresion. Need to reshape the numpy array to format [2,N]###
data_1_2 = np.array([list_data[0],list_data[1]])
data_1_2_reshape = np.reshape(data_1_2, (113965,2))
#### linear regresion module in scipy takes as input a single np array and output a tupple with slope, y-intercept r and more, all collected in res###
res = stats.linregress(data_1_2_reshape)
plt.scatter(list_data[0], list_data[1], s=0.1)
####interesting way to plot a line given slope and y-intercept: multiply X-range by slope and add y-intercept###
plt.plot(list_data[0], res.intercept + res.slope*list_data[0], 'r')
plt.show()

