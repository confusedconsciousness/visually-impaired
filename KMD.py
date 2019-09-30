import keras
from keras import backend as K 
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image 
from keras.models import Model 
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np 

###############################
# KMDNet
###############################
img_h = 256
img_w = 256
img_c = 3
classes = 8 # 3 + 5 (left, mid, right, car, table, chair, person, bed)
x = Input(shape=(img_h, img_w, img_c))
# first convolution block
conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='valid', name='conv1')(x)
conv1 = BatchNormalization(axis=3, name='bn1')(conv1)
conv1 = Activation(activation='relu', name='act1')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

# second convolution block
conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='valid', name='conv2')(pool1)
conv2 = BatchNormalization(axis=3, name='bn2')(conv2)
conv2 = Activation(activation='relu', name='act2')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

# third convolution block
conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid', name='conv3')(pool2)
conv3 = BatchNormalization(axis=3, name='bn3')(conv3)
conv3 = Activation(activation='relu', name='act3')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

# fourth convolution block
conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='valid', name='conv4')(pool3)
conv4 = BatchNormalization(axis=3, name='bn4')(conv4)
conv4 = Activation(activation='relu', name='act4')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

# fifth convolution block
conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid', name='conv5')(pool4)
conv5 = BatchNormalization(axis=3, name='bn5')(conv5)
conv5 = Activation(activation='relu', name='act5')(conv5)
pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

globalPool = GlobalAveragePooling2D(name="globalPool")(pool5)
dense1 = Dense(units=512, activation='relu', name='act6')(globalPool)
pred = Dense(units=classes, activation='sigmoid', name='output')(dense1)

model = Model(inputs=x, outputs=pred)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()









