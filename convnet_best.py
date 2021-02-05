# Load imports
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  Flatten, MaxPool2D
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, BatchNormalization


# Load the data

path = 'D:/PycharmProjects/data_comp_2/ift3395-6390-quickdraw/train.npz'
path_test = 'D:/PycharmProjects/data_comp_2/ift3395-6390-quickdraw/test.npz'

# load and split the training data into train and val
with np.load(path) as data:
    X_train, X_val, Y_train, Y_val = train_test_split(data['arr_0'], data['arr_1'], test_size=0.33)
    # Reshape data.
    X_train = X_train.reshape(1005, 28, 28, 1)
    X_val = X_val.reshape(495, 28, 28, 1)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')

    # Data normalization
    mean = np.mean(X_train, axis=(0))
    std = np.std(X_train, axis=(0))
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # one-hot encode the labels to produce a softmax on the last layer
    nb_classes = 6
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_val = np_utils.to_categorical(Y_val, nb_classes)

# Create a checkpoint for the best weights, based on the best validation accuracy

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Begin the creation of the Convnet model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# generate variations of images with 4 different manners
datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1,rotation_range=15)

training_generator = datagen.flow(X_train, Y_train, batch_size=8,seed=7)

history = model.fit_generator(training_generator,steps_per_epoch=(len(X_train))//8,
                              epochs=250, validation_data=(X_val, Y_val),
                              validation_steps=(len(X_train))//8, callbacks=callbacks_list)

# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- PREDICTION OF TEST DATA ----------------------------------------------
# have to recreate the model, and then at the end we load the best weights that were saved earlier

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

# load the best weights
model.load_weights("weights.best.hdf5")


# ------------------ Use the best model to predict the new classes of unseen data ----------------

with np.load(path_test) as data:
  X_test = data['arr_0']

# Reshape data.
  X_test = X_test.reshape(60000, 28,28,1)
  X_test = X_test.astype('float32')

# normalize the data
  X_test = (X_test - mean) / std

# prediction
ynew = model.predict_classes(X_test)

#  Create CSV file
import pandas as pd
df = pd.DataFrame(data=ynew, columns=["Category"])
df['Id'] = [i for i in range(len(df.Category))]
columns_titles = ["Id", "Category"]
df = df.reindex(columns=columns_titles)
df.to_csv('submission_ift_6390.csv', index=False)