import numpy as np
import healpy as hp
import pandas as pd
import fitsio
import pickle
import os
import h5py
import matplotlib.pyplot as plt
from map import *
from astropy.io import fits

"""

"""

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'
nside = 256

"""
Load all data.
"""

with open(data_dir+'/crosscorr_fwhm_xi.pickle', 'rb') as handle:
	xi = pickle.load(handle)
	
with open(data_dir+'/crosscorr_fwhmvarxi.pickle', 'rb') as handle:
	varxi = pickle.load(handle)
	
with open(data_dir+'/crosscorr_fwhm_weights_xi.pickle', 'rb') as handle:
	xi_weights = pickle.load(handle)
	
with open(data_dir+'/crosscorr_fwhm_weightsvarxi.pickle', 'rb') as handle:
	varxi_weights = pickle.load(handle)
	
print(xi)
print(varxi)


plt.errorbar(x=np.arange(10,100,9), y=xi, yerr=varxi, label='Without weights')
plt.errorbar(x=np.arange(10,100,9), y=xi_weights, yerr=varxi_weights, label = 'With weights')
plt.legend()
plt.show()


