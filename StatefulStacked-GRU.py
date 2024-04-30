'''
Created on May 25, 2019

@author: Ashesh
'''
from math import sqrt
import numpy, math, keras
from numpy import array
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
from keras.layers.recurrent import GRU
from keras.regularizers import L1L2
from keras import optimizers
from keras.callbacks import LearningRateScheduler
 
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
numpy.random.seed(2)

# load the dataset
dataframe = read_csv('ghi/sourceFile/GHI_October_Orissa.csv', header=0, index_col=0)

# Converting dataframe into ndarray    
dataset = dataframe.values
# ensure all data are float
dataset = dataset.astype('float64')

'''
Making the dataset suitable for equal division w.r.t batch-size for stateful architecture model
'''
totalRows = numpy.size(dataset, axis=0)
# To delete unwanted last 8 row from dataset to make the size suitable for stateful LSTM model
keepRows = totalRows - 8
dataset = numpy.delete(dataset, slice(keepRows, totalRows), axis=0)


# selecting only the irradiance values
dhiValues = dataset[:,[0]]
dhiValues = dhiValues.astype('float64')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset)

# specify the number of lag hours (n_hours), features, predicted output timesteps (n_out), batch-size 
n_hours = 48
n_features = 6
n_out = 24
#batch_size = 32 # batch-size for 30 days of month
batch_size = 52 # batch-size for 31 days of month

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
test = numpy.delete(test, -1, axis=0) # deleting unwanted last row of test data
print("Train length:", len(train), "Test length:", len(test))
'''
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
'''
# split into input and outputs
n_obs = n_hours * n_features
n_target_obs = n_out * n_features
train_X = train[:, :n_obs]
train_y = train[:, -n_target_obs:]  # train y with all the features of n_out
test_X = test[:, :n_obs]
test_y = test[:, -n_target_obs:]    # test y with all the features' of n_out
train_y = train_y[:, :1]            # train y with only irradiance feature of first n_out step
test_y = test_y[:, :1]              # test y with only irradiance feature of first n_out step

# Concatenating the irradiance values of (n_out-1) steps to train y and test y
temp = -n_target_obs
for i in range(n_out-1):
    temp = temp + n_features
    train_y = numpy.concatenate((train_y, train[:, [temp]]), axis=1)

print("train_y.shape:", train_y.shape) 

temp = -n_target_obs
for i in range(n_out-1):
    temp = temp + n_features
    test_y = numpy.concatenate((test_y, test[:, [temp]]), axis=1)
  
print("train_X shape: ", train_X.shape[0], " ", train_X.shape[1])

# reshape input to be 3D form [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print(train_X.shape[1], " ", train_X.shape[2])

# design network
model = Sequential()    # define the sequential model
model.add(GRU(10, return_sequences=True, 
              batch_input_shape=(batch_size, train_X.shape[1], train_X.shape[2]), stateful=True)) # setting the batch-input shape and 1st layer of the network
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(GRU(10, stateful=True))   # 2nd layer of the network
model.add(Dense(n_out))             # output layer of the network

# summarize model layers
print("Model summary-", model.summary())

'''
setting hyperparameters with optimizers, learning rate, decay rate etc.
'''
adam = optimizers.Adam(lr=0.00)

def step_decay(epoch):
    initial_lrate = 0.003
    drop = 0.6
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
# to compile the model
model.compile(loss='mean_squared_error', optimizer=adam)

# fit network
for i in range(100):
    print("Epoch: ", i+1)
    history = model.fit(train_X, train_y, epochs=1, callbacks=callbacks_list,
                    validation_data=(test_X, test_y), batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

# make predictions
trainPredict = model.predict(train_X, batch_size=batch_size)
testPredict = model.predict(test_X, batch_size=batch_size)
'''
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
'''
# invert predictions
scaler.fit_transform(dhiValues)
trainPredict = scaler.inverse_transform(trainPredict)
train_y = scaler.inverse_transform(train_y)
testPredict = scaler.inverse_transform(testPredict)
test_y = scaler.inverse_transform(test_y)


# calculate root mean squared error
trainScore = sqrt(mean_squared_error(train_y, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = sqrt(mean_squared_error(test_y, testPredict))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dhiValues)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[n_out:len(trainPredict)+n_out, :] = trainPredict[:, [0]]

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dhiValues)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+n_out+1:len(dataset)+n_out, :] = testPredict[:, [0]]

# plot baseline and predictions
plt.plot(dhiValues, label='Actual')
plt.plot(trainPredictPlot, label='train')
plt.plot(testPredictPlot, label='test')
plt.title('Hour Vs GHI for Orissa(October) - StackedStatefulGRU')
plt.xlabel('Hour')
plt.ylabel('GHI (w/m^2)')
plt.legend(loc='upper right')
plt.show()

'''
plt.plot(test_y[0])
plt.plot(testPredict[:,0])
plt.show()
'''
