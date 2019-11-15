import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
from scipy import stats
plt.switch_backend("Agg")

#data_dir = utils.dat_dir()
data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/'
#graph_dir = utils.fig_dir()
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs/'

with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)

with open(data_dir+'/ngal_norm.pickle', 'rb') as handle:
	ngal_norm = pickle.load(handle)

def plot_hist(pixel_data):
	'''
	returns a multi-panel panel plot with each panel showing 
	the histogram of a systematic parameter
	'''
	cols = [x for x in pixel_data.columns if 'fraction' not in x]
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
	plt.savefig(graph_dir+"sys_hist.png")	    

	return None

def plot_corr(pixel_data):
	'''
	returns a correlation matrix 
	of the systematic parameters
	input: pixel_data
	'''
	df = pd.DataFrame(pixel_data, columns = pixel_data.columns) #convert to a panda dataframe
	df = df[[x for x in pixel_data.columns if 'fraction' not in x]]
	corr = df.corr() #compute the correlation matrix

	plt.figure(figsize=(10, 10))

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

	plt.savefig(graph_dir+"sys_corr.png")	    

	return None

def plot_ngal(ngal_norm, pixel_data, nbins, percut, average_mode = 'median'):
	'''
	returns a multipanel figure, with each panel 
	showing the trend between the normalized 
	gal number density and a systematic parameters
	Inputs: 
	  ngal_norm = normalaized ngal,
	  averge_mode = if 'mean', then the mean density in each bin is computed
		        if 'median', then the median density in each bin is computed
	  	        default: 'median'
	  pixel_data = a dataframe with the same number of rows as 
	  ngals, each column correspond to a systematic parameter;
	  nbins + number of bins for making the plots
	  percut = a tuple containing the desired lower and upper 
	  percentile cuts to be applied to the systematic parameters
	'''
	cols = [x for x in pixel_data.columns if 'fraction' not in x]
	ncols = len(cols)

	nr, nc = len(cols)/3, 3
	fig , axs = plt.subplots(nrows= nr, ncols= nc , sharex= False, sharey= False, figsize= (15,10))
	cnt = 0
	for i in range(nr):
		for j in range(nc):

			y = ngal_norm
			x = pixel_data[cols[cnt]]

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

			#tt= pixel_data.copy()
			#tt = tt[mask]
			#tt['binnumber'] = binnumber #Add the binnumber to every pixel
			#qq = tt.groupby('binnumber').fraction.mean().reset_index(name='frac_means') #Group by binnumber and take mean fraction

			#bin_means_frame = pd.DataFrame({'binnumber':np.arange(1,nbins), 'bin_means':bin_means}) #Create bin means frame for merging

			#kk = pd.merge(qq, bin_means_frame, how='left', on = 'binnumber') #Merge fraction means and bin means on binnumber
			#kk['corrected_bin_means'] = kk.bin_means / kk.frac_means #Correct the bin means

			#Take the corrected binmeans to be the new binmeans
			#bin_means = kk.sort_values(by='binnumber', ascending=True).corrected_bin_means.values 
			#print(bin_means)

			# Compute the uncertainty of y in each bin							     
			bin_errors, bin_edges, binnumber = stats.binned_statistic(x[mask],
									     y[mask], 
									     statistic = stats.sem,
									     bins = bins)
			# Compute the bin centers for plotting
			bin_centers = .5*(bin_edges[1:]+bin_edges[:-1])
			axs[i,j].errorbar(bin_centers, bin_means, bin_errors, fmt = "o", capsize  = 2)
			axs[i,j].tick_params(axis='both', which='major', labelsize=13.5)

			axs[i,j].set_xlabel(cols[cnt], fontsize=18)
			axs[i,j].set_ylabel(r"$n_{\rm gal}/\bar{n}_{\rm gal}$", fontsize=18)

			cnt+=1

	fig.suptitle("average mode = "+str(average_mode)+"percentile cuts="+str(percut[0])+","+str(percut[1]), fontsize=16)
	#plt.title("average mode = "+str(average_mode)+"percentile cuts="+str(percut[0])+","+str(percut[1])) 
	plt.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.savefig(graph_dir+"sys_ngal.png")	    
        

plot_hist(pixel_data)  
plot_corr(pixel_data)  
plot_ngal(ngal_norm, pixel_data, nbins=5, percut = [2, 98], average_mode = "median")
