import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
This code explores the data in the dataset. The Pandas DataFrame containing the parameter values of
the pixels and the normalized galaxy density array are read from the data directory. 
Histograms are plotted of different sets of parameters in the DataFrame. 
"""

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'


with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)

with open(data_dir+'/ngal_norm.pickle', 'rb') as handle:
	ngal_norm = pickle.load(handle)

print('Parameters: {}'.format(pixel_data.columns))


def plot_ngal(X,Y):


	cols = X.columns
	ncols = len(cols)

	nr, nc = len(cols)/3, 3
	fig , axs = plt.subplots(nrows= nr, ncols= nc , sharex= False, sharey= True, figsize= (15,10))
	cnt = 0
	

	for i in range(nr):
		for j in range(nc):
			y = Y
			x = X[cols[cnt]]

			axs[i,j].scatter(x,y, alpha=0.2, s=4)
			cnt += 1
	
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.show()


all_data = pd.DataFrame(pixel_data).copy()
all_data['ngal_norm'] = ngal_norm

fraction_lim = 0.1
filtered_pixel_data = all_data.loc[all_data.fraction > fraction_lim]


use_cols = [x for x in filtered_pixel_data.columns if x != 'fraction']
use_cols = [x for x in use_cols if x != 'ngal_norm']

#Percentile cut
percentiles_low = np.percentile(filtered_pixel_data[use_cols], 5, axis=0)
percentiles_high = np.percentile(filtered_pixel_data[use_cols], 95, axis=0)

print(percentiles_low)
print(percentiles_high)

perc_df = pd.DataFrame([percentiles_low,percentiles_high], columns = use_cols)

print(perc_df)

tt = filtered_pixel_data[use_cols].apply(lambda x: x[(x>perc_df.loc[0,x.name]) & 
                                    (x < perc_df.loc[1,x.name])], axis=0)


qq = tt.dropna(axis=0, how='all')
print(len(filtered_pixel_data))
print(len(qq))

X = filtered_pixel_data[use_cols]
Y = filtered_pixel_data['ngal_norm'].values

#plot_ngal(X,Y)

'''
2d histogram
'''

def plot_2dhist(X,Y, bins=100):

	cols = X.columns
	ncols = len(cols)

	nr, nc = len(cols)/3, 3
	fig, axs = plt.subplots(nrows=nr, ncols=nc, sharex=False, sharey=True, figsize=(15,10))
	cnt = 0

	for i in range(nr):
		for j in range(nc):
			y = Y
			x = X[cols[cnt]]

			counts, xedges, yedges, im = axs[i,j].hist2d(x,y, bins=bins)
			plt.colorbar(im, ax=axs[i,j])


			axs[i,j].set_xlabel(cols[cnt], fontsize=14)
			axs[i,j].set_ylabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=14)

			cnt += 1

	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.ylim((0,6))
	plt.show()

plot_2dhist(X,Y)

with open(data_dir+'/linregprediction.pickle', 'rb') as handle:
	linreg_pred = pickle.load(handle)



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
				axs[i,j].hist2d(x, Y, bins=(50,50), range = [[xmin, xmax], [ymin, ymax]])

			else:
				axs[i,j].hist2d(x,linreg_pred.flatten(), bins=(50,50), range = [[xmin, xmax], [ymin, ymax]])
			axs[i,j].set_xlabel(cols[cnt], fontsize=12)
		cnt+=1
	#plt.colorbar(im, ax=ax)
	#plt.ylim((0,5))
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	fig.savefig(graph_dir+figname)
	plt.show()

plot_2dpred_5(X, Y, use_cols[0:5], linreg_pred, figname='true_vs_pred_1.png')
plot_2dpred_5(X, Y, use_cols[5:10], linreg_pred, figname='true_vs_pred_2.png')
plot_2dpred_5(X, Y, use_cols[10:], linreg_pred, figname='true_vs_pred_3.png')

#plot_2dpred_5(X, Y, use_cols[0:5], linreg_pred, figname='true_vs_pred_1_ylim5.png')
#plot_2dpred_5(X, Y, use_cols[5:10], linreg_pred, figname='true_vs_pred_2_ylim5.png')
#plot_2dpred_5(X, Y, use_cols[10:], linreg_pred, figname='true_vs_pred_3_ylim5.png')
'''
f, ax = plt.subplots(figsize=(9, 7))
plt.hist(ngal_norm, bins=20, histtype= "step", lw=2, color = 'black')
ax.set_yscale('log')
ax.set_ylim((10**0,10**5))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.tight_layout()
plt.show()
f.savefig(graph_dir+'/ngal_hist.png')

f1, ax1 = plt.subplots(figsize=(9, 7))
plt.scatter(pixel_data['fraction'], ngal_norm, s=15.0, color = 'black')
plt.xlim((0,1))
plt.ylim((0,350))
plt.xlabel(r"$f_{\rm pix}$", fontsize=18)
plt.ylabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()
f1.savefig(graph_dir+'/ngal_vs_fraction.png')

f2, ax2 = plt.subplots(figsize=(9, 7))
plt.scatter(pixel_data['fraction'], ngal_norm, s=15.0, color = 'black')
plt.xlim((0,0.2))
plt.ylim((0,350))
plt.xlabel(r"$f_{\rm pix}$", fontsize=18)
plt.ylabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()
f2.savefig(graph_dir+'/ngal_vs_fraction_zoom.png')

f3, ax3 = plt.subplots(figsize=(9, 7))
plt.hist(pixel_data['fraction'], bins=20, histtype= "step", lw=2, color = 'black')
ax3.set_yscale('log')
ax3.set_ylim((10**0,10**5)) 
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r"$f_{\rm pix}$", fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.tight_layout()
plt.show()
f3.savefig(graph_dir+'/pixel_fraction_hist.png')

f4, ax4 = plt.subplots(figsize=(9, 7))
plt.hist(pixel_data['fraction'], bins=20, histtype= "step", cumulative=True, lw=2, color = 'black')
#ax4.set_yscale('log')
#ax4.set_ylim((10**0,10**5)) 
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel(r"$f_{\rm pix}$", fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.tight_layout()
plt.show()
f4.savefig(graph_dir+'/pixel_fraction_cumulative_hist.png')

'''



