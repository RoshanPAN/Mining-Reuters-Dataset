#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-11 14:59:21
# @Author  : Lowson Pan (pan.692@osu.edu)

import numpy as np
import math
from class_Dataset import Dataset, SingleClassDataset

class TreeNode:
    def __init__(self, isLeaf, class_label, split_attr_idx=None, split_position=None):
        """
        Leaf Node:
            class label: str
            other: None
        Non-leaf Node:
            class label: None
            split_attr_idx: int
            split_position: float
        """
        self.left, self.right = None, None
        self.isLeaf = isLeaf
        if isLeaf:
            self.split_attr_idx, self.split_attr_pivot = None, None
            self.class_label = class_label
        else:
            self.split_attr_idx = split_attr_idx
            self.split_attr_pivot = split_position
            self.class_label = None
        self.InfoGain = 0



class SingleLabelDecisionTreeClassifier:
    def __init__(self, dimension, min_split_entropy_threshold=0.0):
        # self.training_sample_fv = np.array([], dtype=np.float32).reshape(0, dimension)
        # self.training_class_label = []  # list of 0: No, 1: Yes
        self.min_split_entropy_threshold = min_split_entropy_threshold
        self.decision_tree = None
        self.new_id_list = None

    def predict(self, fv):
        treeRoot = self.decision_tree
        cl = self._DecisionTreeSearch(treeRoot, fv)
        return cl

    def _DecisionTreeSearch(self, node, fv):
        # Base Case
        if node.isLeaf:
            return node.class_label
        # Recursive Case
        split_attr_idx, split_attr_pivot = node.split_attr_idx, node.split_attr_pivot
        if fv[split_attr_idx] < split_attr_pivot:
            return self._DecisionTreeSearch(node.left, fv)
        else:
            return self._DecisionTreeSearch(node.right, fv)

    def fit(self, training_fv_list, class_label_list, new_id_list=None):
        # self.training_sample_fv = np.concatenate((self.training_sample_fv, training_feature_vectors), axis =0)
        # self.training_class_labels = training_class_labels
        if new_id_list is None:
            new_id_list = np.arange(len(class_label_list), dtype=np.int32)
            self.new_id_list = new_id_list
        dataset = SingleClassDataset(int(training_fv_list.shape[1]))
        for i, fv in enumerate(training_fv_list):
            dataset.add(new_id_list[i], class_label_list[i], fv)
        used_pivot_attrs = set() # remember which attributes has been used as splitting pivot
        self.decision_tree = self._TreeInduction_ID3(dataset, used_pivot_attrs)
        # del self.training_sample_fv, self.training_class_labels


    def _TreeInduction_ID3(self, dataset, used_pivot_attrs):
        # Base Case 1: Empty dataset
        # One of the child dataset is empty -> create a leaf node for it and mark it as 0, negtive
        if len(dataset.fv_nparray) == 0:
            return TreeNode(isLeaf=True, class_label=0)
        # Base Case 2:  Belong to 1 class only
        # Only have 1 kind of class label -> leaf node
        if len(set(dataset.class_label_list)) == 1:
            return TreeNode(isLeaf=True, class_label=dataset.class_label_list[0])
        # Base Case 3: < Minimum impurity threshold
        entropy = dataset.calShannonEntropy()
        if  entropy < self.min_split_entropy_threshold:
            return TreeNode(isLeaf=True, class_label=dataset.majority_class() )
        # Find best split
        max_InfoGain = -1
        best_split = (-1, None, None, None)
        for attr_idx in xrange(dataset.fv_nparray.shape[1]):
            if attr_idx in used_pivot_attrs:
                continue
            # split numerical value into 3 part: [ avg - std, avg, avg + std ]
            for split_position in dataset.numerical_split_position(attr_idx):
                InfoGain, left, right = self._splitDataset_getInfoGain(dataset, attr_idx, split_position)
                if InfoGain > max_InfoGain:
                    best_split = (attr_idx, split_position, left, right)
                    max_InfoGain = InfoGain
        # Create Tree Node and connect left and right sub-tree
        attr_idx, split_position, left, right = best_split
        parent = TreeNode(isLeaf=False, class_label=None, split_attr_idx=attr_idx, split_position=split_position)
        used_pivot_attrs.add(attr_idx)
        parent.left = self._TreeInduction_ID3(left, used_pivot_attrs)
        parent.right = self._TreeInduction_ID3(right, used_pivot_attrs)
        parent.InfoGain = max_InfoGain
        return parent

    def _splitDataset_getInfoGain(self, dataset, attr_idx, split_position):
        parentEnt = dataset.calShannonEntropy()
        # Split dataset into 2 parts
        left, right = SingleClassDataset(dataset.fv_dimension), SingleClassDataset(dataset.fv_dimension)
        for new_id, fv, cl in dataset:
            if fv[attr_idx] < split_position:
                left.add(new_id, cl, fv)
            else:
                right.add(new_id, cl, fv)
        # Calculate Information Gain
        left_Ent, right_Ent = left.calShannonEntropy(), right.calShannonEntropy()
        left_weight, right_weight = float(left.size) / dataset.size, float(right.size) / dataset.size
        InfoGain = parentEnt - left_weight * left_Ent - right_weight * right_Ent
        return InfoGain, left, right


class MultiLabelDecisionTreeClassifier:
    """
    Consists of several single label decision tree classifier
    """
    def __init__(self, fv_dimension, min_split_entropy_threshold=0.0):
        self.fv_dimension = fv_dimension
        self.min_split_entropy_threshold = min_split_entropy_threshold
        self.classifier_list = []

    def fit(self, training_fv_list, cl_indicator_matrix, new_id_list=None):
        if new_id_list is None:
            new_id_list = np.arange(len(training_fv_list), dtype=np.int32)
            self.new_id_list = new_id_list
        # create a single label classifiter for each distinct class label
        cl_dimension = int(cl_indicator_matrix.shape[1])
        for i in xrange(cl_dimension):
            clf = SingleLabelDecisionTreeClassifier(self.fv_dimension, self.min_split_entropy_threshold)
            clf.fit(training_fv_list, cl_indicator_matrix[:, i], new_id_list)
            self.classifier_list.append(clf)

    def predict(self, fv):
        cl_dimension = len(self.classifier_list)
        prediction = np.zeros((1, cl_dimension), dtype=np.int16)
        for i in xrange(cl_dimension):
            clf = self.classifier_list[i]
            prediction[0, i] = clf.predict(fv)
        return prediction



if __name__ == "__main__":
    a = np.array([], dtype=np.float32).reshape(0, 10)
    b = np.arange(10 , dtype=np.float32).reshape(1, 10)
    print a.shape
    print b.shape
    print np.concatenate((a, b), axis=0)
