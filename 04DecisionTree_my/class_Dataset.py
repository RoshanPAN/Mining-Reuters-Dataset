#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-25 20:31:06
# @Author  : Lowson Pan (pan.692@osu.edu)

import random
import os
import numpy as np
import math
import cPickle
from sklearn import tree
from sklearn.preprocessing import MultiLabelBinarizer

class Dataset(object):
    def __init__(self, fv_attrs_list):
        self.size = 0
        self.fv_dimension = len(fv_attrs_list)
        self.sample_id_list = [] # sample's ID
        # Feature Vector
        self.fv_attrs_list = fv_attrs_list[:]
        self.fv_nparray = np.array([], dtype=np.float32).reshape(0, self.fv_dimension)  # storing feature vectors
        # Class Lable
        self.class_label_lists = []
        self.classes_ = []  # all distinct class labels in this dataset

    def add(self, newID, class_label_list, feature_vector):
        self.sample_id_list.append( newID )
        self.fv_nparray = np.concatenate((self.fv_nparray, [ feature_vector ]), axis = 0)
        self.class_label_lists.append( class_label_list[:] )
        self.update_classes_( class_label_list )
        self.size += 1

    def update_classes_(self, class_label_list):
        classes_set = set(self.classes_)
        for label in class_label_list:
            if label in classes_set:
                continue
            self.classes_.append(label)

    def iter_feature_vector(self):
        pass


    def __str__(self):
        s = ""
        s +=  "-" * 20 + "\n"
        s +=  "ID List: " + str(self.sample_id_list)
        s +=  "-" * 20 + "\n\n"
        s +=  "Feature Vectors Array: " + str(self.fv_nparray)
        s +=  "-" * 20 + "\n\n"
        s +=  "Class Labels List: " + str(self.class_label_lists)
        s +=  "-" * 20 + "\n\n"
        return s




class SingleClassDataset(object):
    def __init__(self, dimension):
        self.size = 0
        self.fv_dimension = dimension
        self.sample_id_list = [] # sample's ID
        # Feature Vector
        self.fv_nparray = np.array([], dtype=np.float32).reshape(0, self.fv_dimension)  # storing feature vectors
        # Class Lable
        self.class_label_list = []

    def add(self, newID, class_label, feature_vector):
        self.sample_id_list.append( newID )
        self.fv_nparray = np.concatenate((self.fv_nparray, [ feature_vector ]), axis = 0)
        self.class_label_list.append( class_label )
        self.size += 1

    def calShannonEntropy(self):
        num_samples = float(self.size)
        if num_samples == 0:
            return 0.0
        counter = {1: 0, 0:0}
        for cl in self.class_label_list:
            counter[cl] += 1
        assert len(counter) == 2
        p0, p1 = counter[0] / num_samples, counter[1] / num_samples
        if p0 == 0 or p1 == 0:
            return 0.0
        else:
            Entropy = -1 * ( p0 * math.log(p0, 2) + p1 * math.log(p1, 2) )
            return Entropy


    def majority_class(self):
        cnt_0, cnt_1 = 0, 0
        for cl in self.class_label_list:
            if cl == 0:
                cnt_0 += 1
            elif cl == 1:
                cnt_1 += 1
        assert cnt_0 + cnt_1 == len(self.class_label_list)
        return 1 if cnt_1 > cnt_0 else 0

    def numerical_split_position(self, attr_idx):
        avg = np.average(self.fv_nparray[:, attr_idx])
        std = np.std(self.fv_nparray[:, attr_idx])
        return [ avg - std, avg, avg + std ]

    def __iter__(self):
        for i in xrange(len(self.fv_nparray)):
            yield self.sample_id_list[i], self.fv_nparray[i, :], self.class_label_list[i]

    def __str__(self):
        s = ""
        s +=  "-" * 20 + "\n"
        s +=  "ID List: \n" + str(self.sample_id_list)
        s +=  "-" * 20 + "\n"
        s +=  "Feature Vectors Array: \n" + str(self.fv_nparray)
        s +=  "-" * 20 + "\n"
        s +=  "Class Labels List: \n" + str(self.class_label_list)
        s +=  "-" * 20 + "\n"
        return s
