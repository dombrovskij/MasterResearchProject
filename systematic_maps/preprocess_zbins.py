import numpy as np
import healpy as hp
import pandas as pd
import fitsio
import pickle
import os
import h5py
import matplotlib.pyplot as plt
from map import *

"""
This code loads the systematic maps and the galaxy density map from the data directory.
The galaxy density map was created in galdens_new.py. The KiDS mask is obtained and used on all of the maps.
The galaxy density is normalized using the pixel fraction and average galaxy density. All systematic
maps are put into a Pandas DataFrame and that DataFrame along with the normalized galaxy density array
are pickled to the data directory.
"""

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'
nside = 256

"""
Load all data.
"""
with open(data_dir+'/zbins.pickle', 'rb') as handle:
	zbins = pickle.load(handle)

sysmap_dict = {} #Dictionary to store systematic maps

print('Reading systematic parameter maps...')
for f_name in os.listdir(data_dir+'/maps'): #Go to data directory
	if (f_name.startswith('dr4')) & (f_name.endswith('_new.fits')): #Read only systematic maps
	
		parameter_name = f_name.split('_')[2]
		parameter_array = hp.fitsfunc.read_map(data_dir+'/maps/'+f_name)
		
		sysmap_dict[parameter_name] = parameter_array

		
print('Obtaining KiDS mask...')
pixel_mask, pixel_fraction = good_fraction(nside) #Returns the good pixel mask and the number of good pixels in each pixel

print('Pixel mask shape: {}'.format(pixel_mask.shape))

masked_dataset = {}
for key, value in sysmap_dict.items(): #Mask all systematic maps

	masked_dataset[key] = value[pixel_mask]	
	
#Calculate ratio and pixel fraction

nside_high = 4096
npix_high = hp.nside2npix(nside_high) #total amount of pixels in high resolution

nside_low = nside
npix_low = hp.nside2npix(nside_low) #total amount of pixels in low resolution

ratio = npix_high / npix_low #Amount of pixels from high resolution in low resolution pixel
ratio = float(ratio)

pixel_fraction = pixel_fraction / ratio #Make it an actual fraction

#Modify nstar to account for the pixel fraction
masked_dataset['nstar'] = masked_dataset['nstar'] / pixel_fraction

#Create pandas dataframe in which each row contains all parameters of a single pixel
print('Creating Pandas DataFrame...')
dataset_frame = pd.DataFrame.from_dict(masked_dataset)
dataset_frame['fraction'] = pixel_fraction

#Remove pixels that have a nan value for one or more parameters
rows_containing_nans = dataset_frame.isnull().any(axis=1)
dataset_frame_filtered = dataset_frame[~rows_containing_nans].copy()
print('Removed {} rows from DataFrame containing nans...'.format(rows_containing_nans.sum()))


def preprocess(zbin_dict, show_plot = False):

	print('Preprocessing ngal map {} < z <= {}'.format(zbin_dict['min'], zbin_dict['max']))

	ngal_map = hp.fitsfunc.read_map(data_dir+'/ngal_maps/ngal_'+str(zbin_dict['min']).replace('.','')+'_'+str(zbin_dict['max']).replace('.','')+'.fits') #Read in galaxy density map

	if show_plot:

		#Plot the healpix map
		hp.visufunc.mollview(ngal_map)
		plt.show()
		

	"""
	Mask the data.	
	"""
	ngal_masked = ngal_map[pixel_mask] #Mask ngal
	print('Shape of masked ngal array: {}'.format(ngal_masked.shape))

	"""
	Normalize ngal.
	"""

	#Calculate average galaxy density
	average_ngal = np.sum(ngal_masked) / np.sum(pixel_fraction)
	print('Average NGAL: {}'.format(average_ngal))

	#Calculate the eventual normalized galaxy density per pixel
	#n_i / (f_i * n_avg)
	
	ngal_norm = ngal_masked / pixel_fraction
	ngal_norm = ngal_norm / average_ngal

	
	return ngal_masked, ngal_norm


for k in zbins.keys():

	dataset_frame_filtered_copy = dataset_frame_filtered.copy()

	zbin_min = str(zbins[k]['min']).replace('.','')
	zbin_max = str(zbins[k]['max']).replace('.','')
	
	ngal_masked_zbin, ngal_norm_zbin = preprocess(zbins[k])
	
	#Remove the same indices from the ngal_norm array
	new_ngal_norm_zbin = np.delete(ngal_norm_zbin, rows_containing_nans[rows_containing_nans].index)
	new_masked_ngal_zbin = np.delete(ngal_masked_zbin, rows_containing_nans[rows_containing_nans].index)

	#Save masked ngal
	with open(data_dir+'/ngal_masked_'+zbin_min+'_'+zbin_max+'.pickle', 'wb') as handle:
		pickle.dump(new_masked_ngal_zbin, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print('NUMBER OF ZEROS:')
	print(len(np.where(new_ngal_norm_zbin == 0)[0]))

	print('New Length of data and ngal_norm (must be equal):') #Just to check, they must be equal
	print(len(dataset_frame_filtered_copy))
	print(len(new_ngal_norm_zbin))

	dataset_frame_filtered_copy['ngal_norm'] = new_ngal_norm_zbin
	
	print(dataset_frame_filtered_copy.head())

	#Save the data
	with open(data_dir+'/pixel_data_'+zbin_min+'_'+zbin_max+'.pickle', 'wb') as handle:
		pickle.dump(dataset_frame_filtered_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
	print('Data saved to {}.'.format(data_dir))


