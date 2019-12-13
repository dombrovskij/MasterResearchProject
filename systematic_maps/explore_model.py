import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from filter import *

"""
This code explores a model. The Pandas DataFrame containing the parameter values of
the pixels and the normalized galaxy density are read from the data directory. 
"""

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'


with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)
	
print('Parameters: {}'.format(pixel_data.columns))

temp = fraction_lim(pixel_data, frac_lim=0.1)

use_cols = [x for x in pixel_data.columns if (x != 'fraction') & (x != 'ngal_norm')]		

filtered_pixel_data = percentile_cuts(temp, use_cols, low_cut=5, high_cut=95, verbose=True)	

X = filtered_pixel_data[use_cols].copy()
Y = filtered_pixel_data['ngal_norm'].values


with open(data_dir+'/linregprediction.pickle', 'rb') as handle:
	linreg_pred = pickle.load(handle)


#CHECK WHICH PREDICTIONS THESE ARE... ARE THESE ZEROS??
linreg_pred = linreg_pred.flatten()
print('Number of negative predictions being clipped to 0: {}'.format(len(linreg_pred[linreg_pred < 0])))
linreg_pred[linreg_pred < 0] = 0 #clip negative predictions to 0

def plot_2dpred_5(X, Y, cols, linreg_pred, figname=''):


	ymin = 0
	ymax = np.max([np.max(linreg_pred), np.max(Y)])
	
	nr, nc = 5, 2
	fig, axs = plt.subplots(nrows = nr, ncols = nc, sharey=True, figsize=(9,15))
	axs[0,0].set_title('True', fontsize=18)
	axs[0,1].set_title('Predicted', fontsize=18)
	cnt = 0

	for i in range(nr):
		for j in range(nc):

			x = X[cols[cnt]].values

			xmin = np.min(x)
			xmax = np.max(x)
			if j == 0:
				axs[i,j].hist2d(x, Y, bins=(80,80), range = [[xmin, xmax], [ymin, ymax]])

			else:
				axs[i,j].hist2d(x,linreg_pred.flatten(), bins=(80,80), range = [[xmin, xmax], [ymin, ymax]])
			axs[i,j].set_xlabel(cols[cnt], fontsize=12)
		cnt+=1
	#plt.colorbar(im, ax=ax)
	#plt.ylim((0,5))
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	fig.savefig(graph_dir+'/model_results/'+figname)
	plt.show()

plot_2dpred_5(X, Y, use_cols[0:5], linreg_pred, figname='true_vs_pred_1.png')
plot_2dpred_5(X, Y, use_cols[5:10], linreg_pred, figname='true_vs_pred_2.png')
plot_2dpred_5(X, Y, use_cols[10:], linreg_pred, figname='true_vs_pred_3.png')


