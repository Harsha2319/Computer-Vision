##########################################################################
# HOW TO RUN
#
# - The program takes 2 arguments : train_x file name and train_y file name
# - Train files should be in 'big' folder
# - Test files should be in 'small' folder
#
# Example : python proj2.py "train_x_1_10.csv" "train_y_1_10.csv"
#
##########################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
from numpy.random import seed
import sys

# set the random seeds to make sure your results are reproducible
seed(1)
tf.random.set_seed(1)

# insist on 2 arguments
if len(sys.argv) != 3:
  print(sys.argv[0], "takes 2 arguments. Not ", len(sys.argv)-1)
  print("Arguments: Train_x Train_y. Example: ",
        sys.argv[0]," train_x_1_10.csv train_y_1_10.csv")
  sys.exit()

# specify path to training data and testing data
folderbig = "big"
foldersmall = "small"

train_x_location = foldersmall + "/" + sys.argv[1]
train_y_location = foldersmall + "/" + sys.argv[2]
test_x_location = folderbig + "/" + "test_x.csv"
test_y_location = folderbig + "/" + "test_y.csv"

print("Loading training & testing data")
x_train_2d = np.loadtxt(train_x_location, dtype="uint8")
x_train = x_train_2d.reshape(-1,28,28,1)
y_train = np.loadtxt(train_y_location, dtype="uint8")

x_test_2d = np.loadtxt(test_x_location, dtype="uint8")
x_test = x_test_2d.reshape(-1,28,28,1)
y_test = np.loadtxt(test_y_location, dtype="uint8")

print("Pre processing")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# define the training model
model = keras.Sequential([
    layers.MaxPool2D(4, 4, input_shape=(28,28,1)),
    layers.Conv2D(16, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')])

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("training...")
model.fit(x_train, y_train, epochs=10)

print("evaluate")
model.evaluate(x_test, y_test)
#############################################################################
"""

SAMPLE RESULT

>>>python proj2.py "train_x_1_10.csv" "train_y_1_10.csv"

Loading training & testing data
Pre processing

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
max_pooling2d (MaxPooling2D) (None, 7, 7, 1)           0
_________________________________________________________________
conv2d (Conv2D)              (None, 7, 7, 16)          160
_________________________________________________________________
batch_normalization (BatchNo (None, 7, 7, 16)          64
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 32)          4640
_________________________________________________________________
batch_normalization_1 (Batch (None, 7, 7, 32)          128
_________________________________________________________________
flatten (Flatten)            (None, 1568)              0
_________________________________________________________________
dropout (Dropout)            (None, 1568)              0
_________________________________________________________________
dense (Dense)                (None, 256)               401664
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570
=================================================================
Total params: 409,226
Trainable params: 409,130
Non-trainable params: 96
_________________________________________________________________
training...
Epoch 1/10
188/188 [==============================] - 2s 8ms/step - loss: 2.0268 - accuracy: 0.5629
Epoch 2/10
188/188 [==============================] - 1s 8ms/step - loss: 1.3576 - accuracy: 0.7079
Epoch 3/10
188/188 [==============================] - 1s 8ms/step - loss: 1.2387 - accuracy: 0.7326
Epoch 4/10
188/188 [==============================] - 1s 8ms/step - loss: 1.0720 - accuracy: 0.7715
Epoch 5/10
188/188 [==============================] - 2s 8ms/step - loss: 1.0155 - accuracy: 0.7751
Epoch 6/10
188/188 [==============================] - 2s 8ms/step - loss: 0.9694 - accuracy: 0.7827
Epoch 7/10
188/188 [==============================] - 2s 10ms/step - loss: 0.9051 - accuracy: 0.7912
Epoch 8/10
188/188 [==============================] - 2s 10ms/step - loss: 0.8875 - accuracy: 0.7976
Epoch 9/10
188/188 [==============================] - 2s 9ms/step - loss: 0.8379 - accuracy: 0.8161
Epoch 10/10
188/188 [==============================] - 2s 9ms/step - loss: 0.8070 - accuracy: 0.8216
evaluate
313/313 [==============================] - 1s 3ms/step - loss: 0.8976 - accuracy: 0.7915

>>>
"""