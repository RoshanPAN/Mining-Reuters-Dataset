#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-11 14:59:21
# @Author  : Lowson Pan (pan.692@osu.edu)

from settings import FILTER_COUNT_HIGH, FILTER_COUNT_LOW, MAX_FREQ_IN_FILE, MIN_FREQ_IN_FILE
from math import log

class AllPassages(object):
    def __init__(self, psg_list):
        # calculate total term counts
        self.total_file_num = len(psg_list)
        self.total_term_counter = self.cal_total_freq(psg_list)
        print "[AllPassages] Total term counter generated."
        # calculate total number of words, after filtering
        self.total_word_num = sum(self.total_term_counter.values())
        # calculate the number of tiles a certain term has show up.
        self.term_in_file_counter, self.IDF  = self.cal_file_freq(self.total_term_counter, psg_list)
        print "[AllPassages] Inverted counter generated."

    def cal_total_freq(self, psg_list):
        """
        Merge the frequence counter of all passages from input.
        @passages: list of Passage objects
        @return: dictionary, total frequency of all the file.
        """
        total_term_counter = {}
        for psg in psg_list:
            for word, count in psg.term_counter.items():
                total_term_counter[word] = total_term_counter.get(word, 0) + count
        return total_term_counter

    def cal_file_freq(self, total_term_counter, psg_list):
        term_in_file_counter = {}
        for psg in psg_list:
            for word in psg.get_word_set():
                term_in_file_counter[word] = term_in_file_counter.get(word, 0) + 1
        total_file_num = float(len(psg_list))
        IDF = {}
        for word, cnt in term_in_file_counter.items():
            IDF[word] = log( total_file_num / term_in_file_counter.get(word, 0))
        return term_in_file_counter, IDF

    def gen_feature_vector_attrs(self):
        feature_vector = []
        for word, cnt in self.total_term_counter.items():
            if cnt > FILTER_COUNT_HIGH or cnt < FILTER_COUNT_LOW:
                continue
            term_in_file_cnt = self.term_in_file_counter[word]
            if term_in_file_cnt > MAX_FREQ_IN_FILE or term_in_file_cnt < MIN_FREQ_IN_FILE:
                continue
            feature_vector.append(word)
        print "[AllPassages] Length of Feature Vector: %d" % len(feature_vector)
        return feature_vector

