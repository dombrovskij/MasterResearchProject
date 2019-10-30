import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_dir = '/disks/shear12/dombrovskij/systematic_maps/data'

with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
	pixel_data = pickle.load(handle)

with open(data_dir+'/ngal_norm.pickle', 'rb') as handle:
	ngal_norm = pickle.load(handle)
	
class NeuralNet:

	def __init__(self, x_train, y_train, x_test, y_test):

		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

	def train_evaluate(self, learning_rate=0.01,
	               batch_size=100, n_epoch=100):

		model = Sequential()
		model.add(Dense(10, input_dim=15, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(10, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(10, activation='relu'))


		model.add(Dense(1, activation='relu'))
		
		es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')

		model.compile(loss='mean_squared_error',optimizer='adam',
                      metrics=['mean_squared_error'])

		history = model.fit(self.x_train, self.y_train,
                	batch_size=batch_size,
                	epochs=n_epoch,
                	verbose=1,
                	validation_split=0.1,
			callbacks = [es])

		# evaluate the keras model
		_, MSE = model.evaluate(self.x_test, self.y_test)
		print('MSE: %.2f' % (MSE))
		
X = pixel_data.values
Y = ngal_norm


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

scaler =  StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

scaler =  StandardScaler()
scaled_y_train = scaler.fit_transform(y_train.reshape(-1,1))
scaled_y_test = scaler.transform(y_test.reshape(-1,1))

print(X_train[0])
print(scaled_X_train[0])
print(y_train[0])

test = NeuralNet(scaled_X_train, scaled_y_train, scaled_X_test, scaled_y_test)
test.train_evaluate()
	

