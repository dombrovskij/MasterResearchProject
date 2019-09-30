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

#Plot the healpix map
#hp.visufunc.mollview(ngal_fits)
#plt.show()


print('Obtaining KiDS mask...')
pixel_mask, pixel_fraction = good_fraction(256)

print('Original length ngal array: {}'.format(len(ngal_fits)))
print('Pixel mask shape: {}'.format(pixel_mask.shape))
print('First ten indices of pixel map: {}'.format(pixel_mask[0:10]))\

masked_ngal = ngal_fits[pixel_mask]
print('Shape of masked ngal array: {}'.format(masked_ngal.shape))
