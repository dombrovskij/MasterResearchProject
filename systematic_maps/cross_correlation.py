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
random_DEC_RA ,h = fitsio.read(data_dir+'/new_DR4_9bandnoAwr_uniform_randoms.fits', columns=['DEC','RA'], header=True)	
random_DEC = [x[0] for x in random_DEC_RA]
random_RA = [x[1] for x in random_DEC_RA]

random_map = pd.DataFrame({'DEC':random_DEC, 'RA':random_RA})

	
print(data.head())
print('Len: {}'.format(len(data)))
print(ra_dec_frame.head())
print('Len: {}'.format(len(ra_dec_frame)))
print(random_map.head())
print('Len: {}'.format(len(random_map)))
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

random_map = random_map.sample(frac=0.5, replace=False, random_state=8).copy()
print('Random sample len: {}'.format(len(random_map)))

cross_corr_frame = pd.merge(ra_dec_frame_filtered, data, left_on='ipix', right_on='pixel_idx', how='left')
cross_corr_frame.drop('pixel_idx', axis=1, inplace=True)

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
