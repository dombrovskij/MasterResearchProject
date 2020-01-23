import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from scipy import stats
from filter import *

'''
This code plots a bunch of explorative plots. Elements that are plotted are, amongst others, histogram distributions of systematic parameters, 2D histograms of ngal vs. a systematic parameter, 
a correlation plot, the binned ngal vs. systematic parameter plots.
'''

#data_dir = utils.dat_dir()
data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/'
#graph_dir = utils.fig_dir()
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs/'

with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)
	 
print('Parameters: {}'.format(pixel_data.columns))

temp = fraction_lim(pixel_data, frac_lim=0.1) #Only use pixels with fraction higher than 0.1

use_cols = [x for x in pixel_data.columns if (x != 'fraction') & (x != 'ngal_norm')]		
 
#filtered_pixel_data = percentile_cuts(temp, use_cols, low_cut=5, high_cut=95, verbose=True) #Perform percentile cuts on all use_cols (currently doesn't remove any datapoints)	

X = pixel_data[use_cols].copy()
Y = pixel_data['ngal_norm'].values
Z = pixel_data['fraction'].values

print('Number of zeros: {}'.format(len(np.where(Y==0)[0])))

def plot_hist_single(data, bins=20, lw=2, log=False, ylim = None, cumulative = False, x_label='', y_label='', title=None):

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
	plt.show()
	if title:
		f.savefig(graph_dir+'data_exploration/'+title+'.png')
	
	return None

def plot_hist(pixel_data):

	'''
	returns a multi-panel panel plot with each panel showing 
	the histogram of a systematic parameter
	'''
	
	#cols = [x for x in pixel_data.columns if 'fraction' not in x]
	cols = pixel_data.columns
	ncols = len(cols)

	nr, nc = len(cols)/3, 3
	fig , axs = plt.subplots(nrows= nr, ncols= nc , sharex= False, sharey= False, figsize= (15,10))
	cnt = 0
	for i in range(nr):
		for j in range(nc):

	    		hist = axs[i,j].hist(pixel_data[cols[cnt]], bins= 20, histtype= "step")
	    		axs[i,j].set_xlabel(cols[cnt])
	    		axs[i,j].set_ylabel("counts")
	    		cnt+=1
	plt.tight_layout()
	plt.savefig(graph_dir+"data_exploration/sys_hist.png")	    

	return None
	
def plot_2dhist(X,Y, bins=100, y_lim = None):

	'''
	Returns a multi-panel plot with each panel showing the 2d histogram of a systematic parameter
	with the normalized galaxy density
	'''

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
			#plt.colorbar(im, ax=axs[i,j])


			axs[i,j].set_xlabel(cols[cnt], fontsize=14)
			axs[i,j].set_ylabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=14)

			cnt += 1

	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	
	if y_lim:
		plt.ylim(y_lim)
	plt.savefig(graph_dir+"data_exploration/sys_2dhist.png")	    
	plt.show()
	
	return None
	
def plot_scatter(X,Y, s=15, xlim = None, ylim = None, x_label='', y_label='', title='temp'):

	'''
	Creates scatterplot of Y vs. X. 
	'''

	f, ax = plt.subplots(figsize=(9,7))
	
	plt.scatter(X, Y, s=s, color='black')
	
	if xlim:
		ax.set_xlim(xlim)
		
	if ylim:
		ax.set_ylim(ylim)
	
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	
	plt.xlabel(x_label, fontsize=18)
	plt.ylabel(y_label, fontsize=18)
	
	plt.tight_layout()
	plt.show()
	f.savefig(graph_dir+'data_exploration/'+title+'.png')
	
	return None

def plot_corr(X):


	'''
	returns a correlation matrix 
	of the systematic parameters
	input: pixel_data
	'''
	
	corr = X.corr() #compute the correlation matrix

	# Generate a mask for the upper triangle
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True

	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))

	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 10, as_cmap=True)

	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
	    square=True, linewidths=.5, cbar_kws={"shrink": .5})

	f.savefig(graph_dir+"data_exploration/sys_corr.png")	    

	return None

def plot_ngal(ngal_norm, pixel_data, pixel_fraction, nbins, percut, average_mode = 'median', title=None):

	'''
	returns a multipanel figure, with each panel 
	showing the trend between the normalized 
	gal number density and a systematic parameters
	Inputs: 
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
	
	cols = pixel_data.columns
	ncols = len(cols)

	nr, nc = len(cols)/3, 3
	fig , axs = plt.subplots(nrows= nr, ncols= nc , sharex= False, sharey= False, figsize= (15,10))
	cnt = 0
	for i in range(nr):
		for j in range(nc):

			y = ngal_norm
			x = pixel_data[cols[cnt]]
			z = pixel_fraction

			# Compute the upper and lower percentile cuts
			percs = np.percentile(x, [percut[0], percut[1]])

			# Define a mask based on the percentile cuts
			mask = (x>percs[0])&(x<percs[1])

			# Define the bins at which the mean of y is computed
			bins = np.linspace(x[mask].min(), x[mask].max(), nbins)

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
									     
			#Compute mean of z in each bin
			bin_fmeans, bin_edges, binnumber = stats.binned_statistic(x[mask],
										z[mask],
										statistic = average_mode,
										bins = bins)
			
			#Compute normalization factor
			nghat = np.sum(y[mask])*1./np.sum(z[mask])
			
			#Modify the means and the errorbars
			bin_means = bin_means/nghat/bin_fmeans
			bin_errors = bin_errors/nghat/bin_fmeans
						
			# Compute the bin centers for plotting
			bin_centers = .5*(bin_edges[1:]+bin_edges[:-1])
			
			axs[i,j].plot(bin_centers, bin_means, ls='--', color='C0')
			axs[i,j].errorbar(bin_centers, bin_means, bin_errors, fmt = "o", capsize  = 2, color='C0')
			axs[i,j].tick_params(axis='both', which='major', labelsize=13.5)
			
			axs[i,j].set_xlabel(cols[cnt], fontsize=18)
			axs[i,j].set_ylabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=18)

			cnt+=1

	fig.suptitle("average mode = "+str(average_mode)+"percentile cuts="+str(percut[0])+","+str(percut[1]), fontsize=16)
	#plt.title("average mode = "+str(average_mode)+"percentile cuts="+str(percut[0])+","+str(percut[1])) 
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	
	if title:
		fig.savefig(graph_dir+"data_exploration/"+title+".png")
	else:
		plt.show()
	
	
#--Plot Everything--
'''
print('Plotting plots with whole dataset...')
        
plot_hist_single(pixel_data['ngal_norm'], bins=20, lw=2, log=True, ylim = (10**0, 10**5), x_label=r"$n_{\rm gal}/\bar{n}_{\rm gal}$",
	y_label = 'Count', title='ngal_hist')
	
plot_hist_single(pixel_data.loc[pixel_data['ngal_norm'] < 10].ngal_norm, bins=20, lw=2, log=True, ylim = (10**0, 10**5), x_label=r"$n_{\rm gal}/\bar{n}_{\rm gal}$",
	y_label = 'Count', title='ngal_hist_zoom')
	
plot_hist_single(pixel_data['fraction'], bins=20, lw=2, log=True, ylim = (10**0, 10**5), x_label=r"$f_{\rm pix}$",
	y_label = 'Count', title='pixel_fraction_hist')
	
plot_hist_single(pixel_data['fraction'], bins=20, lw=2, log=True, cumulative=True, ylim = (10**0, 10**5), x_label=r"$f_{\rm pix}$",
	y_label = 'Count', title='pixel_fraction_cumulative_hist')
	

plot_scatter(pixel_data['fraction'], pixel_data['ngal_norm'], s=15, xlim = (0,1), ylim = (0,350),
	 x_label=r"$f_{\rm pix}$", y_label=r"$n_{\rm gal}/\bar{n}_{\rm gal}$", title='ngal_vs_fraction')

plot_scatter(pixel_data['fraction'], pixel_data['ngal_norm'], s=15, xlim = (0,0.2), ylim = (0,350),
	 x_label=r"$f_{\rm pix}$", y_label=r"$n_{\rm gal}/\bar{n}_{\rm gal}$", title='ngal_vs_fraction_zoom')
	 
plot_hist(X)  
plot_2dhist(X,Y, bins=100)
plot_corr(X)  

'''
#plot_ngal(Y, X, Z, nbins=5, percut = [2, 98], average_mode = "mean", title='sys_ngal')
plot_ngal(Y, X, Z, nbins=5, percut = [2, 98], average_mode = "mean")

'''

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
	
	fig_title = 'sys_ngal_'+zbin_min+'_'+zbin_max
	plot_ngal(Y_zbin, X_zbin, Z_zbin, nbins=5, percut = [2,98], title=fig_title)
		
	
'''
