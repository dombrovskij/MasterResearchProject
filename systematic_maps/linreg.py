import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import joblib
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from filter import *

#plt.switch_backend("Agg")

#data_dir = utils.dat_dir()
data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/'
#graph_dir = utils.fig_dir()
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs/'

with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)
	
with open(data_dir+'/zbins.pickle', 'rb') as handle:
	zbins = pickle.load(handle)
	
zbin_data = {}

for k in zbins.keys():

	zbin_min = str(zbins[k]['min']).replace('.','')
	zbin_max = str(zbins[k]['max']).replace('.','')
	
	
	with open(data_dir+'/pixel_data_'+zbin_min+'_'+zbin_max+'.pickle', 'rb') as handle:
		zbin_data[k] = pickle.load(handle)

print('Parameters: {}'.format(pixel_data.columns))

use_cols = [x for x in pixel_data.columns if (x != 'fraction') & (x != 'ngal_norm')]

def filter_data(dataset, frac_lim = 0.1, low_cut=5, high_cut=95):

	temp = fraction_lim(dataset, frac_lim=frac_lim)

	filtered_dataset = percentile_cuts(temp, use_cols, low_cut=low_cut, high_cut=high_cut, verbose=True)

	print(filtered_dataset.ngal_norm.describe())	
	
	return filtered_dataset

def linreg_fit(pixel_data, print_coeff = False, model_title = None):

	filtered_pixel_data = filter_data(pixel_data)

	X = np.array(filtered_pixel_data[use_cols].copy())
	Y = filtered_pixel_data['ngal_norm'].values

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

	print('First ten y_train: {}'.format(y_train[0:10]))

	scalerx =  StandardScaler() #Scale the values
	scaled_X_train = scalerx.fit_transform(X_train)
	scaled_X_test = scalerx.transform(X_test)

	scalery =  StandardScaler()
	scaled_y_train = scalery.fit_transform(y_train.reshape(-1,1))
	scaled_y_test = scalery.transform(y_test.reshape(-1,1))

	print('First ten y_train after scaling {}'.format(scaled_y_train[0:10]))

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(scaled_X_train, scaled_y_train)

	# Make predictions using the testing set
	y_pred = regr.predict(scaled_X_test)
	df = pd.DataFrame.from_dict({'Actual': list(scaled_y_test), 'Predicted': list(y_pred)})
	df1 = df[1000:1010]
	print(df1)

	#Scaled back predictions
	y_pred_scaled_back = scalery.inverse_transform(y_pred)
	df2 = pd.DataFrame.from_dict({'Actual': list(y_test), 'Predicted': list(y_pred_scaled_back)})
	print(df2[1000:1010])

	# The coefficients
	coeff_df = pd.DataFrame(regr.coef_, columns=filtered_pixel_data[use_cols].columns)  
	
	if print_coeff:

		for c in coeff_df.columns:
			print(c)
			print(coeff_df[c])
		
		temp = coeff_df.T.reset_index()
		ax = temp.plot(x='index', y=0, kind='bar', color='grey', legend=None)
		plt.ylim((-1.0, 1.0))
		plt.xticks(fontsize=14, rotation=60, ha='right')
		plt.show()
		
	# The mean squared error
	print("Mean squared error: %.2f"
      	% mean_squared_error(scaled_y_test, y_pred))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % r2_score(scaled_y_test, y_pred))

	#Predict everything:
	full_data_scaled = scalerx.transform(pixel_data[use_cols])
	full_data_predict = regr.predict(full_data_scaled)
	full_predict = scalery.inverse_transform(full_data_predict)
	
	if model_title:

		with open(data_dir+'/model_predictions/'+model_title+'.pickle', 'wb') as handle:
			pickle.dump(full_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
linreg_fit(pixel_data, print_coeff = True, model_title = 'linregprediction')

for k in zbins.keys():

	zbin_min = str(zbins[k]['min']).replace('.','')
	zbin_max = str(zbins[k]['max']).replace('.','')
	model_title = 'linregprediction_'+zbin_min+'_'+zbin_max
	
	linreg_fit(zbin_data[k], print_coeff = True, model_title=model_title)

	

