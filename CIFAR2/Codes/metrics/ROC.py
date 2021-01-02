#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import re
import csv

y_score = []
y_true = []

#Prepare path
inferenced_csv_dir = '../../Results/csv/'
roc_images_dir = '../../Results/ROC/'

#Latest directory
files_path = glob.glob(inferenced_csv_dir + '*')
files_path_sorted = sorted(files_path, key=lambda f: os.stat(f).st_mtime, reverse=True)
most_recent_path= files_path_sorted[0]
df = pd.read_csv(most_recent_path)

#Define scores
y_score = df['1']
y_true = df['label']

#Plot ROC
roc_save_name = roc_images_dir + 'roc.png'

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
auc = metrics.auc(fpr, tpr)
print(auc)
plt.plot(fpr, tpr, label='AUC = %.2f)'%auc, marker='o')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC')
plt.grid()
plt.savefig(roc_save_name)
