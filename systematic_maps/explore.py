import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

"""
This code explores the data in the dataset. The Pandas DataFrame containing the parameter values of
the pixels and the normalized galaxy density array are read from the data directory. 
Histograms are plotted of different sets of parameters in the DataFrame. 
"""

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs'


with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)

with open(data_dir+'/ngal_norm.pickle', 'rb') as handle:
	ngal_norm = pickle.load(handle)

print('Parameters: {}'.format(pixel_data.columns))

fig = plt.figure(figsize=[11,6])
for col in pixel_data.columns:
	if 'lim' in col:
		plt.hist(pixel_data[col], bins=20, alpha=0.5, label=col)
		
plt.xlabel('Magnitude')
plt.ylabel('Count')
plt.legend()
plt.show()
fig.savefig(graph_dir+'/mag_hist.png', dpi=fig.dpi)

fig = plt.figure(figsize=[11,6])
for col in pixel_data.columns:
	if ('lim' not in col) & (col not in ['threshold','BackGr','fwhm']):
		plt.hist(pixel_data[col], bins=20, alpha=0.5, label=col)
		
plt.ylabel('Count')
plt.legend()
plt.show()
fig.savefig(graph_dir+'/hist1.png', dpi=fig.dpi)


fig = plt.figure(figsize=[11,6])
plt.hist(pixel_data['threshold'], bins=20, label='threshold', alpha=0.5)
plt.ylabel('Count')
plt.legend()
plt.show()
fig.savefig(graph_dir+'/threshold_hist.png', dpi=fig.dpi)


fig = plt.figure(figsize=[11,6])
plt.hist(pixel_data['BackGr'], bins=20, label='BackGr', alpha=0.5)
plt.ylabel('Count')
plt.legend()
plt.show()
fig.savefig(graph_dir+'/background_hist.png', dpi=fig.dpi)


fig = plt.figure(figsize=[11,6])
plt.hist(pixel_data['fwhm'], bins=20, label='fwhm', alpha=0.5)
plt.ylabel('Count')
plt.legend()
plt.show()
fig.savefig(graph_dir+'/fwhm_hist.png', dpi=fig.dpi)

