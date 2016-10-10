#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-11 14:59:21
# @Author  : Lowson Pan (pan.692@osu.edu)

"""
Efficiency Metrics - Online Cost & Offline Cost
"""
class MultiLabelClassifierMetricsCalculator:
    def __init__(self):
        self.predictions = []
        self.original_labels = []
        self.all_class_labels = set()
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        self.T, self.F, self.Pos, self.Neg = 0, 0, 0, 0

        self._updated = True

    def addSample(self, original_labels, predicted_class_labels):
        a, b = set(original_labels), set(predicted_class_labels)
        # add original labels into all_class_labels_set
        self.all_class_labels = self.all_class_labels | a
        self.all_class_labels = self.all_class_labels | b
        self.predictions.append(b)
        self.original_labels.append(a)
        # Calculator needs to do preCalculation before returnning any result.
        self._updated = False

    def _preCalculation(self):
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        self.T, self.F, self.Pos, self.Neg = 0, 0, 0, 0
        assert len(self.original_labels) == len(self.predictions)
        i, end = 0, len(self.original_labels)
        N = len(self.all_class_labels)
        while i < end:
            a, b = self.original_labels[i] , self.predictions[i]
            delta_TP = len(a & b)
            delta_FP = len(b - a)
            delta_FN =len(a - b)
            delta_TN = N - delta_TP - delta_FP - delta_FN
            assert ( delta_TP + delta_FP + delta_FN ) == len(a | b)
            self.TP += delta_TP # / float(N)
            self.TN += delta_TN #  / float(N) # Negtive sample, correctly labeled. not in original, and not in predicted
            self.FP += delta_FP # / float(N) # in predicted, but not in original
            self.FN += delta_FN # / float(N)  # not in predicted, but in original.
            i += 1
        self.T = self.TP + self.TN
        self.F = self.FP + self.FN
        self.Pos = self.TP + self.FP
        self.Neg = self.TN + self.FN
        assert (self.Pos + self.Neg) == (self.T + self.F)
        self._updated = True


    def Accuracy(self):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        return float(self.TP + self.TN) / (self.Pos + self.Neg)

    def ErrorRate(self):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        return float(self.FP + self.FN) / (self.Pos + self.Neg)

    def Sensitivity(self):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        return float(self.TP) / (self.Pos + 1)

    def Specificity(self):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        return float(self.TN) / (self.Neg + 1)

    def Precision(self):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        return float(self.TP) / (self.TP + self.FP + 1)

    def Recall(self):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        return float(self.TP) / (self.TP + self.FN + 1)

    def F_Score(self):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        precision = self.Precision()
        recall = self.Recall()
        if precision == 0 or recall == 0:
            return 0
        print precision, recall
        return 2 * precision * recall / (precision + recall )

    def F_Beta_Measure(self, Beta=2):
        if not self._updated:
            self._preCalculation()
            self._updated = True
        p = self.Precision()
        r = self.Recall()
        if p == 0 or r == 0:
            return 0
        return (1 + Beta ** 2) * p * r / (Beta ** 2 * p + r)

    def Accuracy_Percentage_Of_Label(self):
        """
        Percentage of labels predicted correctly.
        """
        if not self._updated:
            self._preCalculation()
            self._updated = True
        total_num_words = sum(map(len, self.original_labels))
        return float(self.TP) / total_num_words

    def Accuracy_Exactly_Same(self):
        cnt, total_num = 0, len(self.original_labels)
        for i in xrange(total_num):
            if self.predictions[i] == self.original_labels[i]:
                cnt += 1
        return float(cnt) / total_num

    def __str__(self):
        if not self._updated:
            self._preCalculation()
        s = "Metrics: \n"
        s += "TP: " + str(self.TP) + "    TN: " + str(self.TN) + "    FP: " + str(self.FP) + "    FN: " + str(self.FN) + "\n"
        s += "True: " + str(self.T) + "    False: " + str(self.F) + "    Negtive: " + str(self.Neg) + "    Positive: " + str(self.Pos) + "\n"
        s += "Accuracy: %.02f,  Accuracy(Pencentage of Label):, %.02f,  Accuracy(Exactly Same): %.02f." \
                    % (self.Accuracy(), self.Accuracy_Percentage_Of_Label(), self.Accuracy_Exactly_Same()) +  "\n"
        s += "ErrorRate: %.02f,  Sensitivity:, %.02f,  Specificity: %.02f, Precision: %.02f" \
                    % (self.ErrorRate(), self.Sensitivity(), self.Specificity(), self.Precision()) +  "\n"
        s += "Recall: %.02f,  F-Score:, %.02f,  F-Beta2-Measure: %.02f." \
                    % (self.Recall(), self.F_Score(), self.F_Beta_Measure()) +  "\n"
        s +=  "\n"
        return s


if __name__ == '__main__':
    cmc = MultiLabelClassifierMetricsCalculator()
    cmc.addSample([1], [1])
    print cmc
    cmc = MultiLabelClassifierMetricsCalculator()
    cmc.addSample([0], [1])
    print cmc
    cmc = MultiLabelClassifierMetricsCalculator()
    cmc.addSample([1, 2], [1, 3])
    print cmc
    cmc = MultiLabelClassifierMetricsCalculator()
    cmc.addSample([1, 2], [1, 3])
    cmc.addSample([1, 2], [1, 3])
    cmc.addSample([1], [1])
    print cmc
