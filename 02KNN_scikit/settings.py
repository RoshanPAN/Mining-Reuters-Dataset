#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-11 14:59:21
# @Author  : Lowson Pan (pan.692@osu.edu)

#　Configuration concerning pickles.
Pickle_Dir = "./data_pickles/"

# For supervised learning, passage without topic should not be included in calculation.
# class_passage, Line 61
Only_Passage_With_Topic = True


"""
Choose which parser you want to use. Currently MY is disabled.
"""
NLTK, MY = 1, 0
parser_setting = NLTK     #NLTK/MY
WEIGHT_TITLE = 3   # Must be integer, if it's more than 1, then title of passage is counted for > 1 times.
"""
2nd Filter.
High Freqency Term Filter: remove a percentage of highest total freqency words including all of the files.
"""
FILTER_COUNT_HIGH = 4500
FILTER_COUNT_LOW = 100
MAX_FREQ_IN_FILE = 4500
MIN_FREQ_IN_FILE = 100
