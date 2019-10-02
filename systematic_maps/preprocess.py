import numpy as np
import healpy as hp
import fitsio
import h5py
import matplotlib.pyplot as plt
from map import *

"""
This code preprocesses the galaxy density map.
"""

ngal_fits = hp.fitsfunc.read_map('ngal.fits') #Read in galaxy density map (created in galdens.py)

nside = 256

#Plot the healpix map
#hp.visufunc.mollview(ngal_fits)
#plt.show()


print('Obtaining KiDS mask...')
pixel_mask, pixel_fraction = good_fraction(nside)

print('Original length ngal array: {}'.format(len(ngal_fits)))
print('Pixel mask shape: {}'.format(pixel_mask.shape))

masked_ngal = ngal_fits[pixel_mask]
print('Shape of masked ngal array: {}'.format(masked_ngal.shape))

#Calculate ratio and pixel fraction

nside_high = 4096
npix_high = hp.nside2npix(nside_high) #total amount of pixels in high resolution

nside_low = nside
npix_low = hp.nside2npix(nside_low) #total amount of pixels in low resolution

ratio = npix_high / npix_low #Amount of pixels from high resolution in low resolution pixel
ratio = float(ratio)

pixel_fraction = pixel_fraction / ratio #Make it an actual fraction

print(pixel_fraction[0:10])


#Calculate average galaxy density

average_ngal = np.sum(masked_ngal) / np.sum(pixel_fraction)

print('Average NGAL: {}'.format(average_ngal))

#Calculate the eventual normalized galaxy density per pixel
#n_i / (f_i * n_avg) 

temp = masked_ngal / pixel_fraction
ngal_norm = temp / average_ngal
