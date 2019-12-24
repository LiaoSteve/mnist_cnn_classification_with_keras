from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# input 1 image : (28,28,1)
model.add(Conv2D(32, kernel_size=(3, 3), 
                 activation='relu',
                 input_shape=input_shape))
# shape(None, 26, 26, 32)
model.add(Conv2D(64, (3, 3), activation='relu'))
# shape(None, 24, 24, 64)
model.add(MaxPooling2D(pool_size=(2, 2)))
# shape(None, 12, 12, 64)
model.add(Dropout(0.25))
# shape(None, 12, 12, 64)
model.add(Flatten())
# now we have 12*12*64 = 9216 inputs
model.add(Dense(128, activation='relu'))
# shape(None, 128)
model.add(Dropout(0.5))
# shape(None, 128)
model.add(Dense(num_classes, activation='softmax'))
# 10 outputs
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('cnn_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model 

model = load_model('cnn_model.h5')

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

digits = ['0','1','2','3','4','5','6','7','8','9']
predict = model.predict(x_test) 

for i in range(5):              
    plt.imshow(x_test[i].reshape(28,28),cmap = 'binary')        
    plt.title('Result:'+ digits[np.argmax(predict[i])])
    plt.show()