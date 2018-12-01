import sys
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU,AveragePooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    TRAIN_CSV_FILE_PATH = sys.argv[1]
    rowsImage, colsImage = 48, 48

    print ('Reading labels and features from training set.')

    X_training = pd.read_csv(TRAIN_CSV_FILE_PATH)
    X_training['feature']


    X_tra = []
    for i in range(len(X_training)):
        temp = X_training['feature'].values[i].split(' ')
        X_tra.append(temp)
        
    X_tra = np.array(X_tra).astype(float)
    X_tra = X_tra/255
    X_tra = X_tra.reshape((len(X_tra),48,48,1))
    y_tra = to_categorical(X_training['label'].values)

    num_classes= y_tra.shape[1]


    datagen = ImageDataGenerator(
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        zoom_range = [0.8, 1.2],
        shear_range = 0.2,
        horizontal_flip = True)
    print ("Generated batches of tensor image data with real-time data augmentation.")
    baseMapNum = 32

    model = Sequential()

    model.add(Conv2D(baseMapNum, (3,3), padding='same', input_shape=(rowsImage, colsImage, 1) ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(2*baseMapNum, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*baseMapNum, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(3*baseMapNum, (3,3), strides=(2, 2),padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(3*baseMapNum, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())


    model.add(AveragePooling2D(pool_size=(3,3)))

    # Fully-connected classifier.
    model.add(Flatten())
    model.add(Dense(units = 128, kernel_initializer = 'glorot_normal'))
    model.add(Dense(num_classes, activation = 'softmax'))
    print ('Created the model.')
    print (model.summary())


    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print ('Compiled the model.')

    fitHistory = model.fit_generator(
        datagen.flow(X_tra, y_tra, batch_size = 128),
        # validation_data=(X_valid, Y_valid), 
        steps_per_epoch = len(X_tra) // 128,
        epochs = 256)

    model.save('GEN_CNN_BEST.h5')

    # Save history of acc to npy file.
    np.save('train_acc_history_GEN_CNN_BEST.npy', fitHistory.history['acc'])

    print ('tra_acc: ', np.amax(fitHistory.history['acc']), 'at epochs = ', np.argmax(fitHistory.history['acc']))

