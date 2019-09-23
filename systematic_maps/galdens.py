import numpy as np
import healpy as hp
import fitsio
import h5py

filename = 'data/shear_dr4.h5'

data = h5py.File(filename, 'r') #read file
print(data.keys())

data_RA = np.array(data['RA'])
data_DEC = np.array(data['DEC'])

data_RA[data_RA > 300] = 360 - data_RA[data_RA > 300] #Correct for 'wrap around'?

nside = 256
npix = hp.nside2npix(nside) #Get total number of pixels in map

theta = (90.0 - data_DEC) * np.pi/180.0 #Convert DEC and RA to angles on the sky
phi = data_RA * np.pi/180.0

ipix = hp.ang2pix(nside, theta, phi, nest=False) #Convert the angles to which pixel it is on the map
bc = np.bincount(ipix, minlength=npix) #returns how many 0, how many 1, how many 2, etc., so essentially counts the objects per pixel
print(np.where(bc != 0)) #Just to check

bc = bc.astype('f16') #Convert int counts to floats

#Below is just a check to make sure it's clear where the number comes from

pixarea = hp.nside2pixarea(nside, degrees=True) #Get the pixel area in deg^2
pixarea_arcmin = pixarea * 60**2 #Convert pixel area in deg^2 to arcmin^2
pixresolution = np.sqrt(pixarea) #The pixel resolution is the square root of the pixel area
print(pixresolution)
print(hp.nside2resol(nside, arcmin=True)) #This should give the same number as the calculation above

bc = bc/pixarea_arcmin #We want the number of objects (galaxies) per arcmin^2 for each pixel
print(bc[0:10]) #Just to see what it looks like
print(bc.shape)

hp.write_map("ngal.fits", bc, overwrite=True) #Save the map
