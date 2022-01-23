import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
import time
import pickle
import keras
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.layers.normalization import BatchNormalization

from PIL import Image,ImageOps
from keras.models import load_model
from keras.preprocessing import image
from numpy import asarray

# load json and create model
json_file = open('cnn2_train_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("cnn2_train_1.h5")
print("Loaded model from disk")


# evaluate loaded model on test data 
# Define X_test & Y_test data first
# loaded_model.compile(loss=losses.mse, optimizer='adam')
#score = loaded_model.evaluate(X_test, Y_test, verbose=0)
#print ("Accuracy : ", score[1]*100)

img = Image.open(".jpg")
print(img.format)
print(img.size)
print(img.mode)

img = ImageOps.grayscale(img)
print(img.format)
print(img.size)
print(img.mode)


img = img.resize((208,208))
print(img.format)
print(img.size)
print(img.mode)

test_img = asarray(img)

res = ['O', 'C', 'E', 'A', 'N']

test_img = np.expand_dims(test_img, axis=2)
test_img = np.expand_dims(test_img, axis=0)
# img_class = loaded_model.predict_classes(img)
#np.argmax(img,axis=1)

ans = loaded_model.predict(test_img)
print(res)
print(ans)

# img = img.reshape((208,208))
plt.imshow(img)
#plt.title(classname)
# plt.show()

# input_img = Input(shape=(208,208,1,))

# fm_size = 16;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)

# fm_size = fm_size*2;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)

# fm_size = fm_size*2;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)

# fm_size = fm_size*2;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)

# fm_size = fm_size*2;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)

# fm_size = fm_size*2;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)

# fm_size = fm_size*2;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)

# fm_size = fm_size*2;
# x = Conv2D(fm_size, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)

# em_size = math.ceil(208/(2**8));

# x = Reshape((em_size*em_size*fm_size,),input_shape=(em_size,em_size,fm_size))(x)
# x=Dense(50,activation='relu', input_shape=(em_size*em_size*fm_size,))(x)
# x=Dense(50,activation='relu')(x)
# x=Dense(10,activation='relu')(x)
# output=Dense(5)(x)


# adam = optimizers.adam(lr=1e-6)
# classifier = Model(input_cla, output)
# classifier.compile(optimizer=adam, loss=losses.mse)

# classifier.summary();




#classifier.load_weights("cnn2_train_1.h5")
#print("Disc Loaded Model ")




# Matriz de confusion Regresor
# Y = ds.data_test['Y'][:];
# tar = np.zeros(Y.shape)
# out = classifier.predict(ds.data_test['X'][:])




