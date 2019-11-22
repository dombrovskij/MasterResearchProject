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

#plt.switch_backend("Agg")

#data_dir = utils.dat_dir()
data_dir = '/disks/shear12/dombrovskij/systematic_maps/data/'
#graph_dir = utils.fig_dir()
graph_dir = '/disks/shear12/dombrovskij/systematic_maps/graphs/'

with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)

with open(data_dir+'/ngal_norm.pickle', 'rb') as handle:
	ngal_norm = pickle.load(handle)

all_data = pixel_data.copy()
all_data['ngal_norm'] = ngal_norm

#Set a fraction limit
fraction_lim = 0.1
filtered_pixel_data = all_data.loc[all_data.fraction > fraction_lim]

print('Number of pixels before filtering: {}'.format(len(all_data)))
print('New number of pixels after filtering: {}'.format(len(filtered_pixel_data)))

use_cols = [x for x in filtered_pixel_data.columns if x != 'fraction']
use_cols = [x for x in use_cols if x != 'ngal_norm']
print('Using columns: {}'.format(use_cols))
X = filtered_pixel_data[use_cols].values
Y = filtered_pixel_data['ngal_norm'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

scalerx =  StandardScaler() #Scale the values
scaled_X_train = scalerx.fit_transform(X_train)
scaled_X_test = scalerx.transform(X_test)

scalery =  StandardScaler()
scaled_y_train = scalery.fit_transform(y_train.reshape(-1,1))
scaled_y_test = scalery.transform(y_test.reshape(-1,1))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(scaled_X_train, scaled_y_train)

# Make predictions using the testing set
y_pred = regr.predict(scaled_X_test)
df = pd.DataFrame.from_dict({'Actual': list(scaled_y_test), 'Predicted': list(y_pred)})
df1 = df.head(25)
print(df1)

#Scaled back predictions
y_pred_scaled_back = scalery.inverse_transform(y_pred)
df2 = pd.DataFrame.from_dict({'Actual': list(y_test), 'Predicted': list(y_pred_scaled_back)})
print(df2.head(25))


# The coefficients
#print('Coefficients: \n', regr.coef_)
coeff_df = pd.DataFrame(regr.coef_, columns=filtered_pixel_data[use_cols].columns)  
print(coeff_df)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(scaled_y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(scaled_y_test, y_pred))

#Predict everything:
full_data_scaled = scalerx.transform(X)
full_data_predict = regr.predict(full_data_scaled)
full_predict = scalery.inverse_transform(full_data_predict)


with open(data_dir+'/linregprediction.pickle', 'wb') as handle:
	pickle.dump(full_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)

