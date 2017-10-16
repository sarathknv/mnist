"""Train a Convolutional Neural Network on the MNIST dataset.

Traditional CNN layer is replaced by MaxMin CNN layer.
Reference: https://github.com/karandesai-96/maxmin-cnn

Dataset was obtained from Kaggle Digit Recognizer challenge.

Optional parameter:
--epochs : Number of epochs
--batch_size : batch size
--plot_model : whether to output an image of the neural network. (bool type)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.utils.vis_utils import plot_model

from maxmin import MaxMin
import argparse

parser = argparse.ArgumentParser(description='Train on MNIST.')
parser.add_argument('--epochs', type=int, default=10, required=False,
                    help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=2048, required=False,
                    help='Batch size.')
parser.add_argument('--plot_model', type=bool, default=False, required=False,
                    help='Plot the network?')
args = parser.parse_args()

# load dataset
train = pd.read_csv('dataset/train.csv').values
test = pd.read_csv('dataset/test.csv').values

# image dimensions
rows = 28
cols = 28

# input: reshape to (None, 28, 28, 1)
x_train = train[:, 1:].reshape(train.shape[0], rows, cols, 1)
x_train = x_train.astype(np.float32)

# output: convert to one hot encoding
y_train = to_categorical(train[:, 0])
y_train = y_train.astype(np.uint8)

# preprocessing
# for faster convergence and to prevent numerical overflows (eg. sigmoid saturates)
x_train = x_train / 255.0

# split dataset into train and validation. (0.1 split)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False)

# our neural network
# where to use batchnorm last answer https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
model = Sequential()
func = 'relu'
layers = [
    MaxMin(16, (3, 3), padding='same', input_shape=(28, 28, 1)),
    Activation('relu'),
    BatchNormalization(),
    MaxMin(16, (3, 3)),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), strides=(2, 2)),
    Dropout(0.25),

    MaxMin(32, (3, 3)),
    Activation('relu'),
    BatchNormalization(),
    MaxMin(32, (3, 3,)),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
]

for l in layers:
    model.add(l)

#model = load_model('saved_models/best_val_acc_epoch_20_bs_4.h5', custom_objects={'MaxMin': MaxMin})

optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = args.epochs
batch_size = args.batch_size

# data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

# callbacks after each epoch
# reduce learning rate if accuracy doesn't change for 3 epochs
decrease_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=0.0000001)
# save model whwenever val acc improves
checkpoint=ModelCheckpoint('saved_models/best_val_acc_epoch_5_bs_256.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                    steps_per_epoch=(x_train.shape[0]/batch_size),
                                    epochs=epochs, validation_data=(x_val, y_val),
                                    verbose=1, callbacks=[decrease_learning_rate, checkpoint])

model.save('saved_models/last_epoch_5_bs_256.h5')

if args.plot_model is True:
    plot_model(model, to_file='mnist_cnn.png', show_shapes=True)


# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].set_title('Loss')
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
ax[0].set_xlabel('epochs')

legend = ax[0].legend(loc='best', shadow=True)

ax[1].set_title('Accuracy')
ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax[1].set_xlabel('epochs')
legend = ax[1].legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
