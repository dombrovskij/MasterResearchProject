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


def plot_ngal(Y, X):


	cols = [x for x in X.columns if 'fraction' not in x]
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

filtered_pixel_data = all_data.loc[all_data.fraction > 0.1]
filtered_pixel_data = filtered_pixel_data.loc[filtered_pixel_data.ngal_norm != 0.0]

use_cols = [x for x in all_data.columns if 'fraction' and 'ngal_norm' not in x]
X = filtered_pixel_data[use_cols]
Y = filtered_pixel_data['ngal_norm'].values

plot_ngal(Y, X)

'''

#Get only magnitude cols
magnitude_cols = []
rest_cols = []

for c in pixel_data.columns:
	if 'lim' in c:
		magnitude_cols.append(c)
	else:
		rest_cols.append(c)

print(magnitude_cols)
print(rest_cols)

#plot_histogram(magnitude_cols, log=True)
#plot_histogram(rest_cols, log=True)
#plot_histogram(['threshold'], log=True)
#plot_histogram(['BackGr'], log=True)
#plot_histogram(['fwhm'], log=True)

percentile_5 = np.percentile(pixel_data, 5, axis=0)
percentile_95 = np.percentile(pixel_data, 95, axis=0)

ylines = dict(zip(pixel_data.columns, zip(percentile_5, percentile_95)))

	
plot_subplots(cols = magnitude_cols, sharex=True, sharey=True)
plot_subplots(cols = rest_cols, sharex=False, sharey=True)

def drop_percentile(cols = [], ylines={}):

	global pixel_data
	new_data = pixel_data.copy()
	
	for col in cols:
	
		to_drop = new_data.loc[(new_data[col] < ylines[col][0]) | (ylines[col][1] < new_data[col])]
		new_data = new_data.drop(to_drop.index).copy()
	
	return new_data
	

pdata_drop_all = drop_percentile(pixel_data.columns, ylines=ylines)

plot_subplots(pcutdata=pdata_drop_all, cols = magnitude_cols, bins=10, sharex=True, sharey=True)
plot_subplots(pcutdata=pdata_drop_all, cols = rest_cols, bins=10, sharex=False, sharey=True)

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





