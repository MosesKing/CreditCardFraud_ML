import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

#
print(tf.__version__)
#
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#
dataframe = pd.read_csv('creditcard.csv')
dataframe.head()
#
dataframe.shape
#
dataframe.isnull().sum()
#
dataframe.info()
#
dataframe['Class'].value_counts()
#
### Balance the Dataset 
#
nonFraud = dataframe[dataframe['Class'] == 0]
fraud = dataframe[dataframe['Class'] == 1]
#
nonFraud.shape, fraud.shape
#
nonFraud = nonFraud.sample(fraud.shape[0])
nonFraud.shape
#
dataframe = fraud.append(nonFraud, ignore_index=True)
dataframe
#
dataframe['Class'].value_counts()
#
x = dataframe.drop('Class', axis=1)
y = dataframe['Class']
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
#
x_train.shape, x_test.shape
#
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
#
x_train.shape
#
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
#
x_train.shape, x_test.shape
#
### Build The CNN ###
#
epochs = 20
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape=x_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(64, 2, activation='relu', input_shape=x_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

#
model.summary()
#
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
#
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)


#
def plot_learningCurve(history, epoch):
    # Plot training & validation accuracy values
    epoch_range = range(1, epoch + 1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model_accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot Training and Validation Loss Values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


#
plot_learningCurve(history, epochs)
#
### Adding MaxPool
#
epochs = 50
model = Sequential()
model.add(Conv1D(32, 2, activation='relu', input_shape=x_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))

model.add(Conv1D(64, 2, activation='relu', input_shape=x_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
#
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
plot_learningCurve(history, epochs)
#

#