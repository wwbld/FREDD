import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
from util import str2int
import numpy as np
import collections
import util

FILE = '../samples/samples_400max_10train_90test_20limit_per_video.csv'

hashmap = {}
count = 0
with open('hashmap.csv') as inf:
    for line in inf:
        currentLine = line.strip().split(",")
        currentLine = list(map(str2int, currentLine))
        hashmap[currentLine[0]] = currentLine[1]
        count += 1
hashmap = collections.OrderedDict(sorted(hashmap.items()))
print("total {0} items".format(count))

model16 = load_model('model_VGG16.h5')
model19_10 = load_model('model_VGG19_10.h5')
model19_19 = load_model('model_VGG19_19.h5')
with open(FILE) as inf:
    count = 0
    related = 0
    vote = 0
    top5 = 0
    for line in inf:
        currentLine = line.strip().split(",")
        currentLine = list(map(str2int, currentLine))
        if currentLine[-1] == 2:
            if count % 1000 == 0 and count != 0:
                print("{0} searched. accu {1}. top5 {1}".format(count, 
                      vote*1.0/related, top5*1.0/related))
            count += 1
            testing_data = np.array(currentLine[0:62208])
            testing_data = np.reshape(testing_data, (1,144,144,3))
            testing_target = currentLine[62208:62209][0]

            prediction16 = model16.predict(testing_data)
            prediction19_10 = model19_10.predict(testing_data)
            prediction19_19 = model19_19.predict(testing_data)           

            list16 = np.argsort(prediction16)[0][::-1][:5]
            list19_10 = np.argsort(prediction19_10)[0][::-1][:5]
            list19_19 = np.argsort(prediction19_19)[0][::-1][:5]

            '''
            print('target is {0}'.format(hashmap[testing_target]))
            print('prediction16 is ', list16)
            print('prediction19_10 is ', list19_10)
            print('prediction19_19 is ', list19_19)
            print('vote is ', util.vote(list16, list19_10, list19_19)[0])
            print('vote top_5 are ', util.vote(list16, list19_10, list19_19)[1])
            print('===========================')
            '''
   
            if testing_target in hashmap:
                related += 1
                target = hashmap[testing_target]
                if target == util.vote(list16, list19_10, list19_19)[0]:
                    vote += 1
                if target in util.vote(list16, list19_10, list19_19)[1]:
                    top5 += 1

print("=================")
print('total is {0}, related is {1}'.format(count, related))
print('accuracy is ', vote*1.0/related)
print('top5 accuracy is ', top5*1.0/related)














