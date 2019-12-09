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
The galaxy density map was created in galdens.py. The KiDS mask is obtained and used on all of the maps.
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


ngal_fits = hp.fitsfunc.read_map(data_dir+'/ngal.fits') #Read in galaxy density map (created in galdens.py)


#Plot the healpix map
hp.visufunc.mollview(ngal_fits)
plt.show()

dataset_dict = {} #Dictionary to store systematic maps

for f_name in os.listdir(data_dir): #Go to data directory
	if f_name.startswith('dr4'): #Read only systematic maps
	
		parameter_name = f_name.split('_')[2]
		parameter_array = hp.fitsfunc.read_map(data_dir+'/'+f_name)
		
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

#Plot the healpix map
temp_for_plot = ngal_fits.copy()
mask = np.ones(temp_for_plot.shape, bool)
mask[pixel_mask] = False
temp_for_plot[mask] = 0
hp.visufunc.mollview(temp_for_plot)
plt.show()

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
	
#Make a plot of ngal before it is normalized

plt.hist(masked_ngal, bins = 100)
plt.xlabel('Unnormalized Galaxy Density')
plt.ylabel('Counts')
plt.savefig(graph_dir+'/unnormalized_ngal_hist.png')

plt.hist(masked_ngal, bins = 100)
plt.xlabel('Unnormalized Galaxy Density')
plt.ylabel('Counts')
plt.yscale('log')
plt.savefig(graph_dir+'/unnormalized_ngal_histlog.png')

#Calculate the eventual normalized galaxy density per pixel
#n_i / (f_i * n_avg)
	
ngal_norm = masked_ngal / pixel_fraction
ngal_norm = ngal_norm / average_ngal

#Plot the healpix map
temp_for_plot = ngal_fits.copy()
mask = np.ones(temp_for_plot.shape, bool)
mask[pixel_mask] = False
temp_for_plot[mask] = 0
temp_for_plot[pixel_mask] = temp_for_plot[pixel_mask] / pixel_fraction
#hp.visufunc.mollview(temp_for_plot)
#plt.show()

temp_for_plot[pixel_mask] = temp_for_plot[pixel_mask] / average_ngal
#hp.visufunc.mollview(temp_for_plot)
#plt.show()

#Modify nstar to account for the pixel fraction
masked_dataset['nstar'] = masked_dataset['nstar'] / pixel_fraction


#Create pandas dataframe in which each row contains all parameters of a single pixel
print('Creating Pandas DataFrame...')
dataset_frame = pd.DataFrame.from_dict(masked_dataset)
dataset_frame['fraction'] = pixel_fraction

#Remove pixels that have a nan value for one or more parameters
rows_containing_nans = dataset_frame.isnull().any(axis=1)
dataset_frame_filtered = dataset_frame[~rows_containing_nans]
print('Removed {} rows from DataFrame containing nans...'.format(rows_containing_nans.sum()))

#Remove the same indices from the ngal_norm array
new_ngal_norm = np.delete(ngal_norm, rows_containing_nans[rows_containing_nans].index)

print('NUMBER OF ZEROS:')
print(len(np.where(new_ngal_norm == 0)[0]))

print('New Length of data and ngal_norm:') #Just to check, they must be equal
print(len(dataset_frame_filtered))
print(len(new_ngal_norm))

#Save the data
with open(data_dir+'/pixel_data.pickle', 'wb') as handle:
	pickle.dump(dataset_frame_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(data_dir+'/ngal_norm.pickle', 'wb') as handle:
	pickle.dump(new_ngal_norm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('Data saved to {}.'.format(data_dir))

