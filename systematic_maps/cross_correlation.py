import numpy as np
import healpy as hp
import pandas as pd
import fitsio
import pickle
import os
import h5py
import matplotlib.pyplot as plt
import treecorr
from map import *
import time

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'
nside = 256

"""

"""

#data_dir = utils.dat_dir()
data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/'
#graph_dir = utils.fig_dir()
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs/'

#Read frame containing systematic parameters, ngal_norm and ipix
with open(data_dir+'/cross_correlation_frame.pickle', 'rb') as handle:
	data = pickle.load(handle)

#Read frame containing RA, DEC and ipix
with open(data_dir+'/RA_DEC_FRAME.pickle', 'rb') as handle:
	ra_dec_frame = pickle.load(handle)
	
	
#Load the model predictions (weights)
with open(data_dir+'/model_predictions/linregprediction.pickle', 'rb') as handle:
	linreg_pred = pickle.load(handle)
linreg_pred = linreg_pred.flatten()

# read both data and header
#random_DEC_RA ,h = fitsio.read(data_dir+'/new_DR4_9bandnoAwr_uniform_randoms.fits', columns=['DEC','RA'], header=True)	
#random_DEC = [x[0] for x in random_DEC_RA]
#random_RA = [x[1] for x in random_DEC_RA]

#random_map = pd.DataFrame({'DEC':random_DEC, 'RA':random_RA})

	
print(data.head())
print('Len: {}'.format(len(data)))
print(ra_dec_frame.head())
print('Len: {}'.format(len(ra_dec_frame)))
#print(random_map.head())
#print('Len: {}'.format(len(random_map)))
print(linreg_pred[0:5])
print('Len: {}'.format(len(linreg_pred)))

print("Total unique ipix in original data: {}".format(ra_dec_frame.ipix.nunique()))
print("Total unique ipix in preprocessed data: {}".format(data.pixel_idx.nunique()))
print("Total ipix in preprocessed that are also in original:{}".format(ra_dec_frame.loc[ra_dec_frame.ipix.isin(data.pixel_idx.unique())].ipix.nunique()))


data['ngal_norm_pred'] = linreg_pred
data['weights'] = 1./data['ngal_norm_pred']
print(data.head())

'''
Out of the original 19438 pixel numbers 19076 are left after masking and removing pixels that have a nan
for at least one of the systematic parameters.
'''

#Filter original data to contain only datapoints we are still working with
ra_dec_frame_filtered = ra_dec_frame.loc[ra_dec_frame.ipix.isin(data.pixel_idx.unique())].copy()

#random_map = random_map.sample(frac=0.5, replace=False, random_state=8).copy()
#print('Random sample len: {}'.format(len(random_map)))

cross_corr_frame = pd.merge(ra_dec_frame_filtered, data, left_on='ipix', right_on='pixel_idx', how='left')
cross_corr_frame.drop('pixel_idx', axis=1, inplace=True)

def calculate_autocc(df, sys_name):

	sys_cat = treecorr.Catalog(ra=df['RA'], dec=df['DEC'], ra_units='deg', dec_units='deg', k=df[sys_name])
	
	autokk = treecorr.KKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units='arcmin')
	

	print('Start calculating autocorrelation of {}'.format(sys_name))
	start = time.time()
	autokk.process(sys_cat, num_threads=80)         # For auto-correlation.

	autokk.write(data_dir+'/auto_correlations/autocc_'+sys_name+'.fits')

	end = time.time()
	print('Done!')
	print('Time: {}'.format(end-start))


def calculate_cc(df, sys_name, weights=False):

	if weights:
		ngal_cat = treecorr.Catalog(ra=df['RA'], dec=df['DEC'], ra_units='deg', dec_units='deg', k=df['ngal_norm'], w=df['weights'])
	else:
		ngal_cat = treecorr.Catalog(ra=df['RA'], dec=df['DEC'], ra_units='deg', dec_units='deg', k=df['ngal_norm'])

	sys_cat = treecorr.Catalog(ra=df['RA'], dec=df['DEC'], ra_units='deg', dec_units='deg', k=df[sys_name])

	kk = treecorr.KKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units = 'arcmin')

	print('Start calculating correlation...')
	start = time.time()
	kk.process(ngal_cat,sys_cat, num_threads=80)   # Compute the cross-correlation function.
	
	if weights:
		kk.write(data_dir+'/cross_correlations/cc_ngal_'+sys_name+'_weights.fits')
	else:
		kk.write(data_dir+'/cross_correlations/cc_ngal_'+sys_name+'.fits')
		
	end = time.time()
	print('Done!')
	print('Time: {}'.format(end-start))

def error_propagation(X, Y, Z, XVAR, YVAR, ZVAR):

	#f  = X / sqrt(Y*Z)

	std_f = np.sqrt((1.0/(Y*Z)) * XVAR + (X**2/(4.0*Z))*(1.0/Y**3) * YVAR + (X**2/(4.0*Y))*(1.0/Z**3) * ZVAR)

	return std_f
	
def plot_cc(sys_name):

	sys_autocorr = treecorr.KKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units='arcmin')
	sys_autocorr.read(data_dir+'/auto_correlations/autocc_'+sys_name+'.fits')
	print(sys_autocorr.xi)

	ngal_autocorr = treecorr.KKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units='arcmin')
	ngal_autocorr.read(data_dir+'/auto_correlations/autocc_ngal_norm.fits')

	kk_no_weights = treecorr.KKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units = 'arcmin')
	kk_no_weights.read(data_dir+'/cross_correlations/cc_ngal_'+sys_name+'.fits')
	print(kk_no_weights.xi)

	kk_weights = treecorr.KKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units = 'arcmin')
	kk_weights.read(data_dir+'/cross_correlations/cc_ngal_'+sys_name+'_weights.fits')
	print(kk_weights.xi)

	#print(kk_no_weights.varxi)
	#print(np.sqrt(kk_no_weights.varxi))

	#Try error propagation
	# f = <dg sys> / sqrt(<sys sys> <dg dg>)
	std_f_no_weights = error_propagation(kk_no_weights.xi, sys_autocorr.xi, ngal_autocorr.xi, kk_no_weights.varxi, sys_autocorr.varxi, ngal_autocorr.varxi)
	std_f_weights = error_propagation(kk_weights.xi, sys_autocorr.xi, ngal_autocorr.xi, kk_weights.varxi, sys_autocorr.varxi, ngal_autocorr.varxi)

	print(std_f_no_weights)
	print(std_f_weights)

 

	plt.errorbar(x=np.arange(10,100,9), y=kk_no_weights.xi/np.sqrt(sys_autocorr.xi*ngal_autocorr.xi), yerr=std_f_no_weights, label='Without weights')
	plt.errorbar(x=np.arange(10,100,9), y=kk_weights.xi/np.sqrt(sys_autocorr.xi*ngal_autocorr.xi), yerr=std_f_weights, label = 'With weights')
	plt.title('Cross-Correlation NGAL with {}'.format(sys_name))
	plt.legend()
	plt.savefig(graph_dir+'/cross_correlation/cc_'+sys_name)
	plt.show()

#calculate_autocc(cross_corr_frame, 'ngal_norm')
#calculate_autocc(cross_corr_frame, 'fwhm')
#calculate_autocc(cross_corr_frame, 'rlim')
#calculate_autocc(cross_corr_frame, 'ilim')
#calculate_cc(cross_corr_frame, 'fwhm', weights=False)
#calculate_cc(cross_corr_frame, 'fwhm', weights=True)
#calculate_cc(cross_corr_frame, 'ilim', weights=False)
#calculate_cc(cross_corr_frame, 'ilim', weights=True)

plot_cc('rlim')
plot_cc('fwhm')
plot_cc('ilim')
'''

def process_cc(cc_frame, ran_frame, sys_name='fwhm', weights=False):

	if weights:

		gal_cat = treecorr.Catalog(ra=cc_frame['RA'], dec=cc_frame['DEC'], w=cc_frame['weights'], ra_units='deg', dec_units='deg')
	
	else:
		gal_cat = treecorr.Catalog(ra=cc_frame['RA'], dec=cc_frame['DEC'], ra_units='deg', dec_units='deg')
	
	sys_cat = treecorr.Catalog(ra=cc_frame['RA'], dec=cc_frame['DEC'], ra_units='deg', dec_units='deg', k=cross_corr_frame[sys_name])
	ran_cat = treecorr.Catalog(ra=ran_frame['RA'], dec=ran_frame['DEC'], ra_units='deg', dec_units='deg')

	nk = treecorr.NKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units = 'arcmin')
	rk = treecorr.NKCorrelation(nbins=10, min_sep=10, max_sep=100, sep_units='arcmin')

	print('Start calculating correlation...')
	start = time.time()
	nk.process(gal_cat,sys_cat, num_threads=80)   # Compute the cross-correlation function.
	rk.process(ran_cat, sys_cat, num_threads=80)
	xi, varxi = nk.calculateXi(rk)
	end = time.time()
	print('Done!')
	print('Time: {}'.format(end-start))
	
	if weights:
		save_name = sys_name+'_weights'
	else:
		save_name = sys_name_

	with open(data_dir+'/crosscorr_'+save_name+'_xi.pickle', 'wb') as handle:
		pickle.dump(xi, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(data_dir+'/crosscorr_'+save_name+'_varxi.pickle', 'wb') as handle:
		pickle.dump(varxi, handle, protocol=pickle.HIGHEST_PROTOCOL)

process_cc(cross_corr_frame, random_map, sys_name='fwhm')
process_cc(cross_corr_frame, random_map, sys_name='fwhm', weights=True)
'''
