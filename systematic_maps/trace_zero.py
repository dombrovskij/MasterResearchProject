import numpy as np
import healpy as hp
import pandas as pd
import fitsio
import pickle
import os
import h5py
import matplotlib.pyplot as plt
from map import *

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'
nside = 256

"""
This code traces back a zero (galaxy density) pixel to its RA and DEC pixel-range and checks whether there
are indeed zero objects in that pixel in the original data.
"""

filename = 'data/shear_dr4.h5'

data = h5py.File(filename, 'r') #read file
print(data.keys())


data_RA = np.array(data['RA']).byteswap().newbyteorder()
data_DEC = np.array(data['DEC']).byteswap().newbyteorder()

data_RA[data_RA > 300] = data_RA[data_RA > 300] - 360 #Correct for 'wrap around'


ngal_fits = hp.fitsfunc.read_map(data_dir+'/ngal_maps/ngal_new.fits') #Read in galaxy density map (created in galdens.py)


print('Obtaining KiDS mask...')
pixel_mask, pixel_fraction = good_fraction(nside) #Returns the good pixel mask and the number of good pixels in each pixel
print(pixel_mask[0:10])
print(pixel_fraction[0:10])
	
print('Pixel mask shape: {}'.format(pixel_mask.shape))

masked_ngal = ngal_fits[pixel_mask] #Mask ngal
print('Shape of masked ngal array: {}'.format(masked_ngal.shape))

traceback_pixel_idx = np.where(masked_ngal == 0)[0][4444] #Randomly choose this pixel to be traceback pixel
mask_idx = pixel_mask[traceback_pixel_idx]
print(mask_idx)
print(ngal_fits[mask_idx])

def IndexToDecRa(index):
	theta,phi=hp.pixelfunc.pix2ang(256,index)
	return -np.degrees(theta-np.pi/2.), np.degrees(np.pi*2.-phi)
	
DEC_zeropix, RA_zeropix = IndexToDecRa(mask_idx) #Get RA and DEC location of zero pixel on 4096 map

pixsize_arcmin = hp.pixelfunc.nside2resol(256, arcmin=True)
print('Pixel size arcmin: {}'.format(pixsize_arcmin))
pixsize_degree = (1./60.) * pixsize_arcmin
print('Pixel size degrees: {}'.format(pixsize_degree))

pix_min = DEC_zeropix - (pixsize_degree/2.)
pix_max = DEC_zeropix + (pixsize_degree/2.)

pix_left = RA_zeropix - (pixsize_degree/2.)
pix_right = RA_zeropix + (pixsize_degree/2.)

print(pix_min, pix_max, pix_left, pix_right)

data_frame = pd.DataFrame({'RA':data_RA, 'DEC':data_DEC})
print(data_frame.head())

obj_in_pix = data_frame.loc[(data_frame.RA > pix_left) & (data_frame.RA < pix_right) & (data_frame.DEC < pix_max) & (data_frame.DEC > pix_min)]

print(len(obj_in_pix))
first_RA, first_DEC = obj_in_pix.iloc[0].values

print(first_RA, first_DEC)

theta_test = (90.0 - first_DEC) * np.pi/180.0
phi_test = first_RA * np.pi/180.0

ipix = hp.ang2pix(256, theta_test, phi_test, nest=False)
print(ipix)

def DecRaToIndex(decl,RA):
    return hp.pixelfunc.ang2pix(256,np.radians(-decl+90.),np.radians(360.-RA))
    
print(DecRaToIndex(first_DEC, first_RA))

print(data_frame.loc[data_frame.RA > 360])

print(np.radians(-first_DEC+90.),np.radians(360.-first_RA))

print((90.0 - first_DEC) * np.pi/180.0, first_RA * np.pi/180.0)
