# -*- coding: utf-8 -*-
# author: @peggy4444


from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import asarray
from sklearn.preprocessing import StandardScaler
rom keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from keras.layers import TimeDistributed





"""# Environment maker"""



# possession defenition / generate state:

def End_list_index_maker(event_columns):
    End_list_index=[]
    for index, row in event_columns.iterrows():
        if (event_columns['team_id'].iloc[index] != event_columns['team_id'].shift(-1).iloc[index]) and (event_columns['team_id'].iloc[index] != event_columns['team_id'].shift(-2).iloc[index]) and ((event_columns['team_id'].iloc[index] == event_columns['team_id'].shift(2).iloc[index]) or (event_columns['team_id'].iloc[index] == event_columns['team_id'].shift(1).iloc[index])) :
            End_list_index.append(index)
    return (End_list_index)

End_list= End_list_index_maker(event_columns)

def possession_list_maker(event_features, End_list):
    all_possessions=[]
    for delays in range(0,len(End_list)-1):
        possession=[]
        df= event_features.iloc[End_list[delays-1]+1:End_list[delays]+1]
        pos= df.to_numpy()
        all_possessions.append(pos)
    return all_possessions

all_possessions_StateI= possession_list_maker(feature_set1, End_list)
all_possessions_StateII= possession_list_maker(feature_set2, End_list)
all_possessions_StateIII= possession_list_maker(feature_set3, End_list)


all_possessions_StateI_array= np.array(all_possessions_StateI).T
all_possessions_StateII_array= np.array(all_possessions_StateIII).T
all_possessions_StateIII_array= np.array(all_possessions_StateIII).T


scaler = StandardScaler()
# transform data
scaled_possessionsI = scaler.fit_transform(all_possessions_StateI_array)
scaled_possessionsII = scaler.fit_transform(all_possessions_StateII_array)
scaled_possessionsIII = scaler.fit_transform(all_possessions_StateIIII_array)



# generate actions (labels= ending action of each possession)

def label_list_maker(event_columns, End_list):
    possessions_label=[]
    for i in End_list:
        labels= event_columns['action_type'].iloc[i]
        possessions_label.append(labels)
    return possessions_label

possession_labels= label_list_maker(event_columns, End_list)

print(len(possession_labels),len(all_possessions))




def in_out(all_possessions_array, possession_label_array):

    X_train, X_test, y_train, y_test = train_test_split(all_possessions_array, possession_label_array)

    X_train = sequence.pad_sequences(X_train, maxlen=10)
    X_test = sequence.pad_sequences(X_test, maxlen=10)

    y_train= to_categorical(y_train)
    y_test= to_categorical(y_test)

    return X_train, X_test, y_train, y_test



print(X_train.shape, y_train.shape)


def model_CNN_LSTM():
    
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(10,25)))
    model.add(MaxPooling1D(pool_size=1))
    model.add(LSTM(100))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=20, batch_size=64)
    # # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(model.summary())
    # # Final evaluation of the model
    print("accuracy: %.2f%%" % (scores[1]*100))

    return model



def model_LSTM():
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(10,25),activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=20, batch_size=64)
    # # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(model.summary())
    # # Final evaluation of the model
    print("accuracy: %.2f%%" % (scores[1]*100))

    return model


def model_3D_CNN():

    X_train = X_train.reshape(10,10,10,3)
    X_test =X_test.reshape(10,10,10,3)
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(10,10,10,3)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.5))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    model.summary()
    # Fit data to model
    history = model.fit(X_train, y_train,
            batch_size=128,
            epochs=40,
            verbose=1,
            validation_split=0.3)

    # Generate generalization metrics
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Plot history: Categorical crossentropy & Accuracy
    plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
    plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
    plt.plot(history.history['acc'], label='Accuracy (training data)')
    plt.plot(history.history['val_acc'], label='Accuracy (validation data)')
    plt.title('Model performance for 3D MNIST Keras Conv3D example')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

    return model




def model_encode_decode_LSTM():
    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(25, 10, 4), return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(4, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train LSTM
    for epoch in range(20):
	# fit model for one epoch on this sequence
	model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
    yhat = model.predict(X_train, batch_size=25, verbose=0)
    
    return model








    













