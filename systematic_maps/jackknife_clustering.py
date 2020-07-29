import numpy as np
import healpy as hp
import pandas as pd
import fitsio
import os
import pickle
import treecorr
from map import *
import time
import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from kmeans import segment_data
from sklearn.decomposition import PCA
import multiprocessing as mp

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'

def jackknife(c):
	
	with fitsio.FITS('/disks/shear13/dombrovskij/jackknife_galaxies/set'+str(c)+'.fits') as fits:
    		gal = fits[1].read()

	with fitsio.FITS('/disks/shear13/dombrovskij/jackknife_randoms/set'+str(c)+'.fits') as fits:
    		rand = fits[1].read()	

	gal_cat = treecorr.Catalog(ra=gal['RA'], dec=gal['DEC'], ra_units='deg', dec_units='deg')
	ran_cat = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], ra_units='deg', dec_units='deg')

	nn = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')
	rr = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')
	dr = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')

	nn.process(gal_cat, gal_cat, num_threads = 20)
	rr.process(ran_cat, ran_cat, num_threads = 20)
	dr.process(gal_cat, ran_cat, num_threads = 20)

	auto_old, varxi = nn.calculateXi(rr, dr)
	theta = nn.meanr

	#With weights

	gal_cat = treecorr.Catalog(ra=gal['RA'], dec=gal['DEC'], w=gal['weights'], ra_units='deg', dec_units='deg')

	nn = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')
	rr = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')
	dr = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')

	nn.process(gal_cat, gal_cat, num_threads=20)
	rr.process(ran_cat, ran_cat, num_threads=20)
	dr.process(gal_cat, ran_cat, num_threads=20)

	gal_cat.clear_cache()
	ran_cat.clear_cache()

	auto_new, varxi = nn.calculateXi(rr, dr)
	theta = nn.meanr


	return (c, theta, auto_old, auto_new)

def jackknife_only_old(gal, random, weight=True):
	
	if weight:
		gal_cat = treecorr.Catalog(ra=gal['RA'], dec=gal['DEC'], w=gal['weights'], ra_units='deg', dec_units='deg')
	else:
		gal_cat = treecorr.Catalog(ra=gal['RA'], dec=gal['DEC'],ra_units='deg', dec_units='deg')

	ran_cat = treecorr.Catalog(ra=random['RA'], dec=random['DEC'], ra_units='deg', dec_units='deg')

	nn = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')
	rr = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')
	dr = treecorr.NNCorrelation(min_sep=10, max_sep=100, nbins=10, sep_units='arcmin')

	nn.process(gal_cat, gal_cat, num_threads = 20)
	rr.process(ran_cat, ran_cat, num_threads = 20)
	dr.process(gal_cat, ran_cat, num_threads = 20)

	auto_old, varxi = nn.calculateXi(rr, dr)
	theta = nn.meanr

	return theta, auto_old
'''

results = []
for c in range(75,100):
	print(c)
	result = jackknife(c)
	results.append(result)

print(results)
print('Done, saving file')
fits = fitsio.FITS('/disks/shear13/dombrovskij/jackknife_galaxies/full_jackknife_results4.fits','rw')
c = np.array([x[0] for x in results])
theta = np.array([x[1] for x in results])
old_auto = np.array([x[2] for x in results])
new_auto = np.array([x[3] for x in results])
array_list=[c, theta, old_auto, new_auto]
names=['c','theta','auto_old', 'auto_new']
fits.write(array_list, names=names)
fits.close()
'''

NJK = 100
auto_old_samples = []
auto_new_samples = []

for i in range(1,5):

	with fitsio.FITS('/disks/shear13/dombrovskij/jackknife_galaxies/full_jackknife_results'+str(i)+'.fits') as fits:
		temp = fits[1].read()

	for old, new in zip(temp['auto_old'], temp['auto_new']):
		auto_old_samples.append(old)
		auto_new_samples.append(new)


#print(auto_new_samples)
auto_old_samples = np.array(auto_old_samples)
auto_new_samples = np.array(auto_new_samples)

auto_old_errors = np.diag(((NJK-1)**2./NJK)*np.cov(auto_old_samples.T))**.5
auto_new_errors = np.diag(((NJK-1)**2/NJK)*np.cov(auto_new_samples.T))**.5

print(auto_old_errors)
print(auto_new_errors)


with open(data_dir+'/clustering_new_error.pickle', 'wb') as handle:
    pickle.dump(auto_new_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(data_dir+'/clustering_old_error.pickle', 'wb') as handle:
    pickle.dump(auto_old_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)


