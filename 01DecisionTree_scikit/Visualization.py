#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-09-11 14:59:21
# @Author  : Lowson Pan (pan.692@osu.edu)

from matplotlib import pyplot as plt

def dict_visualize(dictionary, title, xlabel, ylabel, text):
    plt.figure(1)
    tmp = []
    for key, val in dictionary.items():
        tmp.append((key, val))
    tmp.sort(key=lambda x: x[1], reverse=True)
    x = map(lambda x: x[0], tmp)
    y = map(lambda x: x[1], tmp)
    # print x
    # print y
    # print type(x[0]), type(y[0])
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(  len(y) / 7 * 6, y[0] * 0.95, text)
    plt.show()

def list_visualize(lst, title=" ", xlabel=" ", ylabel=" ", text=" ", sort=True):
    plt.figure(1)
    if sort is True:
        lst.sort(reverse=True)
    # print x
    # print y
    # print type(x[0]), type(y[0])
    plt.plot(lst)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(  len(lst) / 7 * 6, lst[0] * 0.95, text)
    plt.text(  len(lst) / 7 * 6, lst[0] * 0.8, "Sum: %d" %sum(lst))

    plt.show()
