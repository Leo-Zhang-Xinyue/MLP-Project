#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:23:01 2019

@author: leopold
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_file(fname):
    with open (fname+'quora_duplicate_questions.tsv') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd[:10]:
            print(row)
load_file('')

