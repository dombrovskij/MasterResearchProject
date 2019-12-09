import numpy as np
import healpy as hp
import fitsio
import h5py
import matplotlib.pyplot as plt

filename = 'data/shear_dr4.h5'

data = h5py.File(filename, 'r') #read file
print(data.keys())


data_RA = np.array(data['RA'])
data_DEC = np.array(data['DEC'])


#Removed, was in old version
#data_RA[data_RA > 300] = 360 - data_RA[data_RA > 300] #Correct for 'wrap around'?

nside = 256
npix = hp.nside2npix(nside) #Get total number of pixels in map

print('Total number of pixels in 4096 resolution: {}'.format(hp.nside2npix(4096)))
print('Total number of pixels in 256 resolution: {}'.format(npix))
print('Total number of objects: {}'.format(len(data_RA)))
print('4096 resolution avg obj per pixel: {}'.format(float(len(data_RA))/float(hp.nside2npix(4096))))
print('256 resolution avg obj per pixel: {}'.format(float(len(data_RA))/float(npix)))

theta = (90.0 - data_DEC) * np.pi/180.0 #Convert DEC and RA to angles on the sky
phi = (360.- data_RA) * np.pi/180.0

ipix = hp.ang2pix(nside, theta, phi, nest=False) #Convert the angles to which pixel it is on the map
bc = np.bincount(ipix, minlength=npix) #returns how many 0, how many 1, how many 2, etc., so essentially counts the objects per pixel

print('These two numbers should match:')
print('Total number of objects in ngal_new.fits: {}'.format(np.sum(bc)))
print('Total number of objects in catalogue: {}'.format(len(data_RA)))
bc = bc.astype('f16') #Convert int counts to floats

pixarea = hp.nside2pixarea(nside, degrees=True) #Get the pixel area in deg^2
pixarea_arcmin = pixarea * 60**2 #Convert pixel area in deg^2 to arcmin^2
pixresolution = np.sqrt(pixarea_arcmin) #The pixel resolution is the square root of the pixel area

print('These two numbers should match:')
print('Calculated pixel resolution: {}'.format(pixresolution))
print('Pixel resolution given by healpy: {}'.format(hp.nside2resol(nside, arcmin=True))) #This should give the same number as the calculation above

bc = bc/pixarea_arcmin #We want the number of objects (galaxies) per arcmin^2 for each pixel

hp.visufunc.mollview(bc)
plt.show()

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
hp.write_map(data_dir+"/ngal_new.fits", bc, overwrite=True) #Save the map
