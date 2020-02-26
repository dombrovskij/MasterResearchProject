import numpy as np
import healpy as hp
import fitsio
import h5py
import matplotlib.pyplot as plt
import pickle
import pandas as pd

'''
This code creates the galaxy density map from the RA and DEC data of the galaxies, as well as the
galaxy density maps for five redshift bins.
'''

filename = 'data/shear_dr4.h5'
data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/'

data = h5py.File(filename, 'r') #read file
print(data.keys())

data_RA = np.array(data['RA'])
data_DEC = np.array(data['DEC'])
data_z = np.array(data['Z_B'])

print('Number of datapoints: {}'.format(len(data_RA)))

'''
Divide the data into the five redshift bins and create ngal map for each.
'''

print('Minimum redshift in data: {}'.format(np.min(data_z)))
print('Maximum redshift in data: {}'.format(np.max(data_z)))

z_bins = [0.0,0.1,0.3,0.5,0.7,0.9,1.2, np.max(data_z)]
z_digitized = np.digitize(data_z, z_bins, right=True)

zbin_dict = {}

for bin_num in range(1,6):
	
	bin_z = data_z[np.where(np.array(z_digitized) == bin_num+1)] #digitize starts counting at 1
	bin_RA = data_RA[np.where(np.array(z_digitized) == bin_num+1)]
	bin_DEC = data_DEC[np.where(np.array(z_digitized) == bin_num+1)]
	zbin_dict[bin_num] = {'min':z_bins[bin_num], 'max':z_bins[bin_num+1], 
		'RA_values':bin_RA, 'DEC_values':bin_DEC, 'z_values':bin_z}

print('Data divided in 5 redshift bins:')

for k in zbin_dict.keys():
	
	dict_entry = zbin_dict[k]
	print('Bin {} min {}, max {}, n_values {}'.format(k, dict_entry['min'], dict_entry['max'], len(dict_entry['z_values'])))
	print(dict_entry['z_values'][0:10])

def create_ngal_map_zbin(map_dict, show_plot = False):

	sRA = map_dict['RA_values']
	sDEC = map_dict['DEC_values']
	sz = map_dict['z_values']

	print('Creating ngal map min {} < z <= {}...'.format(map_dict['min'], map_dict['max']))

	sRA[sRA > 300] = sRA[sRA > 300] - 360 #Correct for 'wrap around', RA between 300 and 360 get mapped to range -60 to 0.
	
	nside = 256
	npix = hp.nside2npix(nside) #Get total number of pixels in map
	
	print('Number of objects in this map: {}'.format(len(sRA)))
	
	theta = (90.0 - sDEC) * np.pi/180.0 #Convert DEC and RA to angles on the sky
	phi = sRA * np.pi/180.0
	
	ipix = hp.ang2pix(nside, theta, phi, nest=False) #Convert the angles to which pixel it is on the map
	bc = np.bincount(ipix, minlength=npix) #returns how many 0, how many 1, how many 2, etc., so essentially counts the objects per pixel

	pixarea = hp.nside2pixarea(nside, degrees=True) #Get the pixel area in deg^2
	pixarea_arcmin = pixarea * 60**2 #Convert pixel area in deg^2 to arcmin^2
	
	bc = bc/pixarea_arcmin #We want the number of objects (galaxies) per arcmin^2 for each pixel
	
	if show_plot:
		hp.visufunc.mollview(bc)
		plt.show()
		
	data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/ngal_maps'
	hp.write_map(data_dir+'/ngal_'+str(map_dict['min']).replace('.','')+'_'+str(map_dict['max']).replace('.','')+'.fits', bc, overwrite=True) #Save the map
	
for k in zbin_dict.keys():

	create_ngal_map_zbin(zbin_dict[k], show_plot=False)
	del zbin_dict[k]['z_values']
	del zbin_dict[k]['RA_values']
	del zbin_dict[k]['DEC_values']


data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/'	
with open(data_dir+'/zbins.pickle', 'wb') as handle:
	pickle.dump(zbin_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
Create the ngal map for all the data.
'''

data_RA[data_RA > 300] = data_RA[data_RA > 300] - 360 #Correct for 'wrap around', RA between 300 and 360 get mapped to range -60 to 0.

nside = 256 
npix = hp.nside2npix(nside) #Get total number of pixels in map

theta = (90.0 - data_DEC) * np.pi/180.0 #Convert DEC and RA to angles on the sky
phi = data_RA * np.pi/180.0

ipix = hp.ang2pix(nside, theta, phi, nest=False) #Convert the angles to which pixel it is on the map
bc = np.bincount(ipix, minlength=npix) #returns how many 0, how many 1, how many 2, etc., so essentially counts the objects per pixel

'''
Create Pandas dataframe
'''

RA_DEC_Frame = pd.DataFrame({'RA':data_RA,'DEC':data_DEC,'theta':theta,'phi':phi,'ipix':ipix})

with open(data_dir+'RA_DEC_FRAME.pickle', 'wb') as handle:
	pickle.dump(RA_DEC_Frame, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
Continue creating ngal map
'''
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

#hp.visufunc.mollview(bc)
#plt.show()

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/ngal_maps'
hp.write_map(data_dir+"/ngal_new.fits", bc, overwrite=True) #Save the map
