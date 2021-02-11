#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 01:01:47 2020

@author: hiroaki_ikeshita
"""


import pandas as pd
import sys
import json

args = sys.argv

SAVE_FILE_NAME = args[1]    
NUM_FILES = len(args[2:])

file_path = "SETTINGS_FULL.json"

with open(file_path, "r") as json_file:
    jsn = json.load(json_file)
    
    
for i, acc_pred_file in enumerate(args[2:]):
    submission = pd.read_csv(jsn["ACC_SUB_DIR"] + acc_pred_file)
    if i == 0:
        blend_sub = submission
    else:
        blend_sub.iloc[:, 1:] = blend_sub.iloc[:, 1:] + submission.iloc[:, 1:]

blend_sub.iloc[:, 1:] /= NUM_FILES

blend_sub.to_csv(jsn["ACC_SUB_DIR"] + SAVE_FILE_NAME, index=False)
