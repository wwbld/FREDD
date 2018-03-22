import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.applications import VGG16, VGG19, InceptionResNetV2, Xception
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import util
import csv
import collections

FILE = '../samples/samples_400max_10train_1test_20limit_per_video.csv'

SIZE = 144
batch_size = 64
num_classes = 410
epochs = 50

print('getting training data\n')
training_data, training_target, testing_data, testing_target = util.read_csv(FILE)
print("done getting trainng data\n");

hashmap = {}
count = 0
with open('hashmap.csv', 'r') as myfile:
    for line in myfile:
        currentLine = line.strip().split(",")
        currentLine = list(map(util.str2int, currentLine))
        hashmap[currentLine[0]] = currentLine[1]
        count += 1
print("loaded {0} items from old hashmap".format(count))

with open('hashmap.csv', 'a+') as myfile:
    writer = csv.writer(myfile)
    for i in range(len(training_target)):
        if training_target[i] not in hashmap:
            hashmap[training_target[i]] = count
            writer.writerow([training_target[i], count])
            count += 1
        training_target[i] = hashmap[training_target[i]]
    for i in range(len(testing_target)):
        if testing_target[i] not in hashmap:
            hashmap[testing_target[i]] = count
            writer.writerow([testing_target[i], count])
            count += 1
        testing_target[i] = hashmap[testing_target[i]]
print("total items in hashmap is ", count)
hashmap = collections.OrderedDict(sorted(hashmap.items()))
print(hashmap)

training_data = np.reshape(training_data, (-1,SIZE,SIZE,3))
testing_data = np.reshape(testing_data, (-1,SIZE,SIZE,3))

training_target = keras.utils.to_categorical(training_target, num_classes)
testing_target = keras.utils.to_categorical(testing_target, num_classes)

vgg = VGG19(weights='imagenet', include_top=False, input_shape=(SIZE,SIZE,3))
x = vgg.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.7)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=x)

for layer in vgg.layers[:10]:
    layer.trainable = False
for layer in vgg.layers[10:]:
    layer.trainable = True

model.compile(optimizer=optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

print('starting training\n')

history = model.fit(training_data, training_target,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(testing_data, testing_target))
score = model.evaluate(training_data, training_target, verbose=0)
print("training score is ", score)
score = model.evaluate(testing_data, testing_target, verbose=0)
print("testing score is ", score)

model.save('model_VGG19_10.h5') #saving the trained model

