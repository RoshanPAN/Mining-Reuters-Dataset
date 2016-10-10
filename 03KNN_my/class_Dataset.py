#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-25 20:31:06
# @Author  : Lowson Pan (pan.692@osu.edu)

import random
import os
import numpy as np
import cPickle
from sklearn import tree
from sklearn.preprocessing import MultiLabelBinarizer

class Dataset(object):
    def __init__(self, fv_attrs_list):
        self.size = 0
        self.fv_dim = len(fv_attrs_list)
        self.sample_id_list = [] # sample's ID
        # Feature Vector
        self.fv_attrs_list = fv_attrs_list[:]
        self.fv_nparray = np.array([], dtype=np.float32).reshape(0, self.fv_dim)  # storing feature vectors
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
