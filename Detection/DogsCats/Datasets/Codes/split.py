import numpy as np
import pandas as pd

import os
from glob import glob
import datetime

img_size_dir = '../Original/'
#img_size_dir = '../Small/'
csv_dir = img_size_dir + 'Split/'

os.makedirs(csv_dir, exist_ok=True)
df = pd.read_csv(img_size_dir + 'summary.csv').sort_values(by="image_id", ascending=True)

def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    train['split'] = 'train'
    validate = df.loc[perm[train_end:validate_end]]
    validate['split'] = 'val'
    test = df.loc[perm[validate_end:]]
    test['split'] = 'test'
    return train, validate, test

df_category_dog = df[df['class'] == 'dog']
df_category_cat = df[df['class'] == 'cat']

train_0, validation_0, test_0 = train_validate_test_split(df_category_dog)
train_1, validation_1, test_1 = train_validate_test_split(df_category_cat)

df_concat = pd.concat([train_0, validation_0, test_0, train_1, validation_1, test_1])

dt_now = datetime.datetime.now()
dt_name = dt_now.strftime('%Y-%m-%d-%H-%M')

df_concat.to_csv(csv_dir + 'split_' + dt_name + '.csv', index=False)
