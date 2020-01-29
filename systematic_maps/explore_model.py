import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from filter import *

"""
This code explores a model. The Pandas DataFrame containing the parameter values of
the pixels and the normalized galaxy density is read from the data directory. 
"""

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'


with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)
	
print('Parameters: {}'.format(pixel_data.columns))

temp = fraction_lim(pixel_data, frac_lim=0.1) #Use only pixels with a fraction higher than 0.1

use_cols = [x for x in pixel_data.columns if (x != 'fraction') & (x != 'ngal_norm')]		

#filtered_pixel_data = percentile_cuts(temp, use_cols, low_cut=5, high_cut=95, verbose=True) #Do percentile cuts on use_cols (currently does not remove any datapoints)	

#X = filtered_pixel_data[use_cols].copy()
#Y = filtered_pixel_data['ngal_norm'].values

X = pixel_data[use_cols].copy()
Y = pixel_data['ngal_norm'].values
Z = pixel_data['fraction'].values

#Load the model predictions
with open(data_dir+'/model_predictions/linregprediction.pickle', 'rb') as handle:
	linreg_pred = pickle.load(handle)

linreg_pred = linreg_pred.flatten()
print('Number of negative predictions being clipped to 0: {}'.format(len(linreg_pred[linreg_pred < 0])))
linreg_pred[linreg_pred < 0] = 0 #clip negative predictions to 0

print('Max Y: {}'.format(np.max(Y)))
print('Max linreg: {}'.format(np.max(linreg_pred)))

def plot_hist_single(data, bins=20, lw=2, log=False, ylim = None, cumulative = False, x_label='', y_label='', figname=None):

	'''
	Plot a single histogram of the input data (should contain only 1 parameter).
	'''

	f, ax = plt.subplots(figsize=(9,7))
	
	plt.hist(data, bins=bins, histtype='step', cumulative=cumulative, lw=lw, color='black')
	
	if log:
		ax.set_yscale('log')
		
	if ylim:
		ax.set_ylim(ylim)
	
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	
	plt.xlabel(x_label, fontsize=18)
	plt.ylabel(y_label, fontsize=18)
	
	plt.tight_layout()
	
	if figname:
		f.savefig(graph_dir+'/model_results/'+figname)
		
	else:
		plt.show()
	
	return None
	
#plot_hist_single(linreg_pred, bins=20, lw=2, log=True, ylim = (10**0, 10**5), x_label=r"$Predicted\ n_{\rm gal}/\bar{n}_{\rm gal}$",
#	y_label = 'Count', figname='predicted_ngal_hist.png')	


def plot_2dpred_5(X, Y, cols, linreg_pred, figname=None):

	'''
	Returns panel plot of 5 rows 2 columns containing 10 2D histograms. Each row has the 2D histogram of the true distribution of ngal vs. a specific parameter, on the right the
	predicted ngal distribution vs. the same predicted parameter.
	'''

	ymin = 0
	ymax = np.max([np.max(linreg_pred), np.max(Y)]) #Get the max. occuring ngal to set matching ranges
	
	nr, nc = 5, 2
	fig, axs = plt.subplots(nrows = nr, ncols = nc, sharey=True, figsize=(9,15)) #Create the subplots
	axs[0,0].set_title('True', fontsize=18) #Set the titles
	axs[0,1].set_title('Predicted', fontsize=18)
	cnt = 0

	for i in range(nr):
		for j in range(nc):

			x = X[cols[cnt]].values

			xmin = np.min(x)
			xmax = np.max(x)
			
			if j == 0: #0th column (left column) is true ngals
				axs[i,j].hist2d(x, Y, bins=(80,80), range = [[xmin, xmax], [ymin, ymax]])

			else: #1st column (right column) is predicted ngals
				axs[i,j].hist2d(x,linreg_pred.flatten(), bins=(80,80), range = [[xmin, xmax], [ymin, ymax]])
			axs[i,j].set_xlabel(cols[cnt], fontsize=12)
		cnt+=1
		
	#plt.colorbar(im, ax=ax)
	#plt.ylim((0,5))
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	
	if figname:
		fig.savefig(graph_dir+'/model_results/'+figname)
	else:
		plt.show()

#plot_2dpred_5(X, Y, use_cols[0:5], linreg_pred, figname='true_vs_pred_1.png')
#plot_2dpred_5(X, Y, use_cols[5:10], linreg_pred, figname='true_vs_pred_2.png')
#plot_2dpred_5(X, Y, use_cols[10:], linreg_pred, figname='true_vs_pred_3.png')

def plot_ngal(ngal_pred, ngal_norm, pixel_data, pixel_fraction, nbins, percut, average_mode = 'median', title=None):

	'''
	returns a multipanel figure, with each panel 
	showing the trend between the normalized 
	gal number density and a systematic parameters and the predicted normalized gal number density
	and the same systematic parameter.
	Inputs: 
	  ngal_pred = predicted normalized ngal,
	  ngal_norm = normalized ngal,
	  averge_mode = if 'mean', then the mean density in each bin is computed
		        if 'median', then the median density in each bin is computed
	  	        default: 'median'
	  pixel_data = a dataframe with the same number of rows as 
	  ngals, each column correspond to a systematic parameter;
	  nbins + number of bins for making the plots
	  percut = a tuple containing the desired lower and upper 
	  percentile cuts to be applied to the systematic parameters
	'''
	
	#cols = [x for x in pixel_data.columns if 'fraction' not in x]\
	cols = pixel_data.columns
	ncols = len(cols)

	nr, nc = len(cols)/3, 3
	fig , axs = plt.subplots(nrows= nr, ncols= nc , sharex= False, sharey= False, figsize= (15,10))
	cnt = 0
	for i in range(nr):
		for j in range(nc):
			
			y_t = ngal_pred
			y = ngal_norm
			x = pixel_data[cols[cnt]]
			z = pixel_fraction
			
			# Compute the upper and lower percentile cuts
			percs = np.percentile(x, [percut[0], percut[1]])

			# Define a mask based on the percentile cuts
			mask = (x>percs[0])&(x<percs[1])

			# Define the bins at which the mean of y is computed
			#bins = np.linspace(x[mask].min(), x[mask].max(), nbins)
			
			syst_cumsum = np.cumsum(np.sort(x) + np.abs(x.min()))
			syst_cumsum /= 1.*syst_cumsum.max()
			fr = np.linspace(percut[0], percut[1], nbins+1) * 0.01
			bins = np.interp(fr, syst_cumsum, np.sort(x))

			# Compute the mean of y in each bin
			bin_means, bin_edges, binnumber = stats.binned_statistic(x[mask],
									     y[mask], 
									     statistic = average_mode,
									     bins = bins)

			# Compute the uncertainty of y in each bin							     
			bin_errors, bin_edges, binnumber = stats.binned_statistic(x[mask],
									     y[mask], 
									     statistic = stats.sem,
									     bins = bins)
									    
			#Compute mean of y/y_t in each bin
			bin_means_corr, bin_edges, binnumber = stats.binned_statistic(x[mask],
										y[mask]/y_t[mask],
										statistic = average_mode,
										bins = bins)
										
			#Compute the uncertaitny of y/y_t in each bin
			bin_errors_corr, bin_edges, binnumber = stats.binned_statistic(x[mask],
										y[mask]/y_t[mask],
										statistic = stats.sem,
										bins = bins)
			
			#Compute mean of z in each bin
			bin_fmeans, bin_edges, binnumber = stats.binned_statistic(x[mask],
										z[mask],
										statistic = average_mode,
										bins = bins)
			
			# Compute the normalization factors
			#nghat_new = np.sum(y[mask]/y_t[mask])*1./np.sum(z[mask])
			#nghat_old = np.sum(y[mask])*1./np.sum(z[mask])
			
			#Modify the means and the errorbars of the raw measurements
			#bin_means = bin_means/bin_fmeans
			#bin_errors = bin_errors/bin_fmeans
			
			#Modify the means and the errorbars of the corrected measurements
			#bin_means_corr = bin_means_corr/bin_fmeans
			#bin_errors_corr = bin_errors_corr/bin_fmeans
			
			# Compute the bin centers for plotting
			bin_centers = .5*(bin_edges[1:]+bin_edges[:-1])
			axs[i,j].plot(bin_centers, bin_means, ls='--', color='C0')
			axs[i,j].errorbar(bin_centers, bin_means, bin_errors, fmt = "o", capsize  = 2, color='C0')
			
			axs[i,j].plot(bin_centers, bin_means_corr, ls='--', color='C1')
			axs[i,j].errorbar(bin_centers, bin_means_corr, bin_errors_corr, fmt='o', capsize=2, color='C1')
			axs[i,j].tick_params(axis='both', which='major', labelsize=13.5)

			axs[i,j].set_xlabel(cols[cnt], fontsize=18)
			axs[i,j].set_ylabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=18)

			cnt+=1

	fig.suptitle("average mode = "+str(average_mode)+"percentile cuts="+str(percut[0])+","+str(percut[1]), fontsize=16)
	#plt.title("average mode = "+str(average_mode)+"percentile cuts="+str(percut[0])+","+str(percut[1])) 
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	
	if title:
		fig.savefig(graph_dir+"/model_results/"+title+".png")
	else:
		plt.show()
	
	
plot_ngal(linreg_pred, Y, X, Z, nbins=5, percut = [2, 98], average_mode = "mean", title='sys_ngal_corr')
#plot_ngal(linreg_pred, Y, X, Z, nbins=5, percut = [2, 98], average_mode = "mean")

#-- Plots for the z-bins --

with open(data_dir+'/zbins.pickle', 'rb') as handle:
	zbins = pickle.load(handle)
	
#zbin_data = {}
print('Plotting plots of zbins...')

for k in zbins.keys():

	zbin_min = str(zbins[k]['min']).replace('.','')
	zbin_max = str(zbins[k]['max']).replace('.','')
	
	
	with open(data_dir+'/pixel_data_'+zbin_min+'_'+zbin_max+'.pickle', 'rb') as handle:
		zbin_data = pickle.load(handle)
	
	X_zbin = zbin_data[use_cols].copy()
	Y_zbin = zbin_data['ngal_norm'].values
	Z_zbin = zbin_data['fraction'].values
	
	#Load the model predictions
	with open(data_dir+'/model_predictions/linregprediction_'+zbin_min+'_'+zbin_max+'.pickle', 'rb') as handle:
		linreg_pred_zbin = pickle.load(handle)

	linreg_pred_zbin = linreg_pred_zbin.flatten()
	print('Number of negative predictions being clipped to 0.0001: {}'.format(len(linreg_pred_zbin[linreg_pred_zbin < 0])))
	linreg_pred_zbin[linreg_pred_zbin < 0] = 0.0001 #clip negative predictions to 0.0001
	
	print('Max Y: {}'.format(np.max(Y_zbin)))
	print('Max linreg: {}'.format(np.max(linreg_pred_zbin)))
	
	plot_hist_single(linreg_pred_zbin, bins=20, lw=2, log=True, ylim = (10**0, 10**5), x_label=r"$Predicted\ n_{\rm gal}/\bar{n}_{\rm gal}$",
	y_label = 'Count', figname='predicted_ngal_hist_'+zbin_min+'_'+zbin_max+'.png')
	
	fig_title = 'sys_ngal_corr_'+zbin_min+'_'+zbin_max
	plot_ngal(linreg_pred_zbin, Y_zbin, X_zbin, Z_zbin, nbins=5, percut = [2,98], title=fig_title)
	#plot_2dpred_5(X_zbin, Y_zbin, use_cols[0:5], linreg_pred_zbin)

