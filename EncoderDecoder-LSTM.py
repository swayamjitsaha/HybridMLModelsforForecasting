'''
Created on June 15, 2019

@author: Ashesh
'''
from math import sqrt
import numpy
from numpy import array
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Activation
from keras.layers import LSTM, Dropout, Input
from datetime import datetime
from keras.layers.recurrent import GRU
from keras.regularizers import L1L2
from keras.callbacks import LearningRateScheduler
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import optimizers, losses
from keras.utils import plot_model
import math, keras
from keras.callbacks import History
from keras.regularizers import l1, l2
import tensorflow as tf

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
       
# fix random seed for reproducibility
#numpy.random.seed(2)

# load the dataset
dataframe = read_csv('ghi/sourceFile/GHI_July_Orissa.csv', header=0, index_col=0)

# Converting dataframe into ndarray      
dataset = dataframe.values
# ensure all data is float
dataset = dataset.astype('float64')

# selecting only the irradiance values
dhiValues = dataset[:,[0]]
dhiValues = dhiValues.astype('float64')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# specify the number of lag hours (n_hours), features, predicted output timesteps (n_out)
n_hours = 24
n_features = 6
n_out = 4
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, n_out)

print("Shape of Supervised data:", reframed.shape)
print("Initial rows of Supervised data:", reframed.head())

'''
Prepare data
'''
# split into train(70%) and test(30%) sets
dataset = reframed.values
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[:train_size, :], dataset[train_size:, :]
print("Train length:", len(train), "Test length:", len(test))
'''
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

for i in range(10):
    print(train[i][0],"", train[i][6],"", train[i][12],"", train[i][18], "", train[i][24])
'''
# split into input and outputs
n_obs = n_hours * n_features
n_target_obs = n_out * n_features
train_X = train[:, :n_obs]
train_y = train[:, -n_target_obs:]  # train y with all the features of n_out
test_X = test[:, :n_obs]
test_y = test[:, -n_target_obs:]    # test y with all the features' of n_out
train_y_dhi = train_y[:, :1]        # train_y_dhi with only irradiance feature of first n_out step
test_y_dhi = test_y[:, :1]          # test_y_dhi with only irradiance feature of first n_out step
'''
for i in range(10):
    print(train_X[i][0],"", train_X[i][6],"", train_X[i][12])
for i in range(10):
    print(train_y_dhi[i][0])
'''    
# Concatenating the irradiance values of (n_out-1) steps to train y and test y
temp = -n_target_obs
for i in range(n_out-1):
    temp = temp + n_features
    train_y_dhi = numpy.concatenate((train_y_dhi, train[:, [temp]]), axis=1)

print("train_y_dhi.shape:", train_y_dhi.shape) 

temp = -n_target_obs
for i in range(n_out-1):
    temp = temp + n_features
    test_y_dhi = numpy.concatenate((test_y_dhi, test[:, [temp]]), axis=1)

print("train_X shape: ", train_X.shape[0], " ", train_X.shape[1])
print("test_y_dhi.shape:", test_y_dhi.shape)

# reshape input to be 3D form [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

# separate target irradiance(n_out) values for train y and test y
train_dhi_1 = train_y_dhi[:, :1]
train_dhi_2 = train_y_dhi[:, 1:2]
train_dhi_3 = train_y_dhi[:, 2:3]
train_dhi_4 = train_y_dhi[:, 3:4]
test_dhi_1 = test_y_dhi[:, :1]
test_dhi_2 = test_y_dhi[:, 1:2]
test_dhi_3 = test_y_dhi[:, 2:3]
test_dhi_4 = test_y_dhi[:, 3:4]
#train_y = train_y.reshape((train_y.shape[0], n_out, 6))
#test_y = test_y.reshape((test_y.shape[0], n_out, 6))
#train_y_dhi = train_y_dhi.reshape((train_y_dhi.shape[0], train_y_dhi.shape[1]*train_y_dhi.shape[2]))
#test_y_dhi = test_y_dhi.reshape((test_y_dhi.shape[0], test_y_dhi.shape[1]*test_y_dhi.shape[2]))

# setting decoder input as zeros for first decoder block
decoder_input_data = numpy.zeros((train_y.shape[0], 1, 1))
decoder_testInput_data = numpy.zeros((test_y.shape[0], 1, 1))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print(train_X.shape[1], " ", train_X.shape[2])

# design network
'''
Encoder LSTM Network part
'''
encoder_inputs = Input(shape=(train_X.shape[1], train_X.shape[2]))
decoder_inputs = Input(shape=(1, 1))
encoder = LSTM(10, return_sequences=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(10, return_state=True)(encoder)
encoder_states = [state_h, state_c]

'''
Decoder LSTM Network part
'''
decoder_lstm = LSTM(10, return_state=True)
decoder, state_h, state_c = decoder_lstm(decoder_inputs,
                    initial_state=encoder_states)
decoder_outputs1 = Dense(1)(decoder)
decoder_states = [state_h, state_c]
decoder_inputs2 = RepeatVector(1)(decoder_outputs1)
decoder, state_h, state_c = decoder_lstm(decoder_inputs2, 
                                         initial_state=decoder_states)
decoder_outputs2 = Dense(1)(decoder)
decoder_states = [state_h, state_c]
decoder_inputs3 = RepeatVector(1)(decoder_outputs2)
decoder, state_h, state_c = decoder_lstm(decoder_inputs3, 
                                         initial_state=decoder_states)
decoder_outputs3 = Dense(1)(decoder)
decoder_states = [state_h, state_c]
decoder_inputs4 = RepeatVector(1)(decoder_outputs3)
decoder, state_h, state_c = decoder_lstm(decoder_inputs4, 
                                         initial_state=decoder_states)
decoder_outputs4 = Dense(1)(decoder)

'''
setting hyperparameters with optimizers, learning rate, decay rate etc.
'''
epochs = 100
#lrate = 0.1
#decay_rate = lrate / epochs

adam = optimizers.Adam(lr=0.0)
def step_decay(epoch):
    initial_lrate = 0.003
    drop = 0.6
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# Define the prediction decoder model
prediction_decoder_model = Model(inputs=[encoder_inputs, decoder_inputs], 
                                 outputs=[decoder_outputs1, decoder_outputs2, decoder_outputs3, decoder_outputs4])
# summarize layers
print("Prediction decoder model summary-", prediction_decoder_model.summary())
# to compile the model
prediction_decoder_model.compile(loss='mean_squared_error', optimizer=adam)

# fit network
lstm_history = prediction_decoder_model.fit([train_X, decoder_input_data], [train_dhi_1, train_dhi_2, train_dhi_3, train_dhi_4], epochs=epochs, 
                        validation_data=([test_X, decoder_testInput_data], [test_dhi_1, test_dhi_2, test_dhi_3, test_dhi_4]),
                        callbacks=callbacks_list, batch_size=64, verbose=2, shuffle=False)

# make predictions
trainPredict = prediction_decoder_model.predict([train_X, decoder_input_data])
testPredict = prediction_decoder_model.predict([test_X, decoder_testInput_data])
trainPredict = numpy.concatenate(trainPredict, axis=1)
testPredict = numpy.concatenate(testPredict, axis=1)
#trainPredict = trainPredict[:, :, :1]
#testPredict = testPredict[:, :, :1]
'''
# reshape output to be 2D [samples, out_timesteps]
trainPredict = trainPredict.reshape((trainPredict.shape[0], trainPredict.shape[1]*trainPredict.shape[2]))
testPredict = testPredict.reshape((testPredict.shape[0], testPredict.shape[1]*testPredict.shape[2]))
train_y_dhi = train_y_dhi.reshape((train_y_dhi.shape[0], train_y_dhi.shape[1]*train_y_dhi.shape[2]))
test_y_dhi = test_y_dhi.reshape((test_y_dhi.shape[0], test_y_dhi.shape[1]*test_y_dhi.shape[2]))
'''
# plot history
plt.plot(lstm_history.history['loss'], label='train')
plt.plot(lstm_history.history['val_loss'], label='test')
plt.title('model loss - Hybrid')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# invert predictions
scaler.fit_transform(dhiValues)
trainPredict = scaler.inverse_transform(trainPredict)
train_y_dhi = scaler.inverse_transform(train_y_dhi)
testPredict = scaler.inverse_transform(testPredict)
test_y_dhi = scaler.inverse_transform(test_y_dhi)

# calculate root mean squared error
trainScore = sqrt(mean_squared_error(train_y_dhi, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = sqrt(mean_squared_error(test_y_dhi, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dhiValues)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[n_out:len(trainPredict)+n_out, :] = trainPredict[:, [0]]
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dhiValues)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+n_out+1:len(dataset)+n_out+1, :] = testPredict[:, [0]]
'''
y = testPredict[-1: , -(n_out-1):]
testPredictLastValues = y.reshape(y.shape[0]*y.shape[1], 1)
testPredictPlot[len(trainPredict)+n_out+1:len(dataset)+(2*n_out), :] = numpy.append(testPredict[:, [0]], testPredictLastValues, axis=0)
'''
# plot baseline and predictions
plt.plot(dhiValues, label='Actual')
plt.plot(trainPredictPlot, label='train')
plt.plot(testPredictPlot, label='test')
plt.title('Hour Vs GHI for October(Orissa) - Encoder-DecoderLSTM model')
plt.xlabel('Hour')
plt.ylabel('GHI (w/m^2)')
plt.legend(loc='upper right')
plt.show()

'''
plt.plot(test_y[0])
plt.plot(testPredict[:,0])
plt.show()
'''
