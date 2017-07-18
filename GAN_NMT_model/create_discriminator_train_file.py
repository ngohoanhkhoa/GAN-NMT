#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:42:08 2017

@author: macbook975
"""


sf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/out_train.en', 'r')
machine_tf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/out_train.fr', 'r')
human_tf = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/out_train.fr', 'r')

train_en = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/train.en', 'w')
train_fr = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/train.fr', 'w')
train_label = open('/Users/macbook975/Documents/Stage/GAN_NMT_model/data_translate/train.label', 'w')

for idx, (sline, mline, hline) in enumerate(zip(sf, machine_tf, human_tf)):
    sline = sline.strip()
    mline = mline.strip()
    hline = hline.strip()

    # Exception if empty line found
    if sline == "" or mline == "" or hline == "":
        continue

    train_en.write(sline + '\n')
    train_en.write(sline + '\n')
    
    train_fr.write(hline + '\n')
    train_fr.write(mline + '\n')
    
    train_label.write('1 0' + '\n')
    train_label.write('0 1' + '\n')