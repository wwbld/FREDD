import tensorflow as tf
import numpy as np
import collections
import random
import csv

from os import listdir
from os.path import isfile, isdir, join


def str2int(s):
    return int(s)

def str2float(s):
    return float(s)

def read_csv(filename):
    training_features = []
    training_labels = []
    testing_features = []
    testing_labels = []
    with open(filename) as inf:
        for line in inf:
            currentLine = line.strip().split(",")
            currentLine = list(map(str2int, currentLine))
            if currentLine[-1] == 1:
                training_features.append(currentLine[0:62208])
                training_labels.append(currentLine[62208:62209][0])
            else:
                testing_features.append(currentLine[0:62208])
                testing_labels.append(currentLine[62208:62209][0])
    return np.array(training_features), np.array(training_labels), \
           np.array(testing_features), np.array(testing_labels)

def vote(l1, l2, l3):
    hashmap = {}
    for i in range(len(l1)):
        if l1[i] not in hashmap:
            hashmap[l1[i]] = 5-i
        else:
            hashmap[l1[i]] = hashmap[l1[i]] + (5-i)
    for i in range(len(l2)):
        if l2[i] not in hashmap:
            hashmap[l2[i]] = 5-i
        else:
            hashmap[l2[i]] = hashmap[l2[i]] + (5-i)
    for i in range(len(l3)):
        if l3[i] not in hashmap:
            hashmap[l3[i]] = 5-i
        else:
            hashmap[l3[i]] = hashmap[l3[i]] + (5-i)
    maxKey = max(hashmap, key=lambda i : hashmap[i])
    top5 = []
    count = 0
    for w in sorted(hashmap, key=hashmap.get, reverse=True):
        while count < 5:
            top5.append(w)
            count += 1
    return maxKey, top5   














