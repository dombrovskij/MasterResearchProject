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

ngal_fits = hp.fitsfunc.read_map(data_dir+'/ngal_maps/ngal_new.fits') #Read in galaxy density map (created in galdens.py)


#Plot the healpix map
hp.visufunc.mollview(ngal_fits)
plt.show()

dataset_dict = {} #Dictionary to store systematic maps

for f_name in os.listdir(data_dir+'/maps/'): #Go to data directory
	if (f_name.startswith('dr4')) & (f_name.endswith('_new.fits')): #Read only systematic maps
	
		parameter_name = f_name.split('_')[2]
		parameter_array = hp.fitsfunc.read_map(data_dir+'/maps/'+f_name)
		
		dataset_dict[parameter_name] = parameter_array
	

"""
Mask all data.
"""

print('Obtaining KiDS mask...')
pixel_mask, pixel_fraction = good_fraction(nside) #Returns the good pixel mask and the number of good pixels in each pixel
print(pixel_mask[0:10])
print(pixel_fraction[0:10])
	
print('Pixel mask shape: {}'.format(pixel_mask.shape))

masked_ngal = ngal_fits[pixel_mask] #Mask ngal
print('Shape of masked ngal array: {}'.format(masked_ngal.shape))
print('Max ngal in masked_ngal: {}'.format(np.max(masked_ngal)))

masked_dataset = {}
for key, value in dataset_dict.items(): #Mask all systematic maps

	masked_dataset[key] = value[pixel_mask]	

"""
Normalize ngal.
"""

#Calculate ratio and pixel fraction

nside_high = 4096
npix_high = hp.nside2npix(nside_high) #total amount of pixels in high resolution

nside_low = nside
npix_low = hp.nside2npix(nside_low) #total amount of pixels in low resolution

ratio = npix_high / npix_low #Amount of pixels from high resolution in low resolution pixel
ratio = float(ratio)

pixel_fraction = pixel_fraction / ratio #Make it an actual fraction
	
#Calculate average galaxy density
average_ngal = np.sum(masked_ngal) / np.sum(pixel_fraction)
print('Average NGAL: {}'.format(average_ngal))

#Calculate the eventual normalized galaxy density per pixel
#n_i / (f_i * n_avg)
	
ngal_norm = masked_ngal / pixel_fraction
ngal_norm = ngal_norm / average_ngal

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

#Remove the same indices from the ngal_norm array
new_ngal_norm = np.delete(ngal_norm, rows_containing_nans[rows_containing_nans].index)

new_masked_ngal = np.delete(masked_ngal, rows_containing_nans[rows_containing_nans].index)

#Save masked ngal
with open(data_dir+'/ngal_maps/masked_ngal.pickle', 'wb') as handle:
	pickle.dump(new_masked_ngal, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('NUMBER OF ZEROS:')
print(len(np.where(new_ngal_norm == 0)[0]))

print('New Length of data and ngal_norm (must be equal):') #Just to check, they must be equal
print(len(dataset_frame_filtered))
print(len(new_ngal_norm))

print('Max ngal in ngal_norm: {}'.format(np.max(new_ngal_norm)))

dataset_frame_filtered['ngal_norm'] = new_ngal_norm

print('NGAL NORM: {}'.format(dataset_frame_filtered.ngal_norm.values[0:10]))

#Save the data
with open(data_dir+'/pixel_data.pickle', 'wb') as handle:
	pickle.dump(dataset_frame_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('Data saved to {}.'.format(data_dir))

#Write dataframe used for cross correlation (same as pixel_data but keeping track of pixel nums)

new_pixel_mask = np.delete(pixel_mask, rows_containing_nans[rows_containing_nans].index)

dataset_cross_correlation = dataset_frame_filtered.copy()
dataset_cross_correlation['pixel_idx'] = new_pixel_mask

with open(data_dir+'/cross_correlation_frame.pickle', 'wb') as handle:
	pickle.dump(dataset_cross_correlation, handle, protocol=pickle.HIGHEST_PROTOCOL)


