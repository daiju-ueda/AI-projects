import numpy as np
import pandas as pd
import os

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

#ディレクトリ定義
origin_dir = '../Original/' # もとの
target_dir = '../Small/' # 保存先

os.makedirs(target_dir + 'Images/', exist_ok=True)

#縮小の縮尺
resize_ratio = 5

#csvの変換
df = pd.read_csv('../Original/summary.csv')
df_resized = pd.read_csv('../Original/summary.csv')
xy = ['width','height','xmin','ymin','xmax','ymax']

for i in xy:
    df_resized[i] = (df_resized[i]/resize_ratio).round()
df_resized.to_csv(target_dir + 'summary.csv', index=False)

#画像の変換
for index, row in df_resized.iterrows():
    img = Image.open(origin_dir + 'Images/' + row['image_id'] + '.jpg')
    img_resize = img.resize((int(row['width']), int(row['height'])))
    img_resize.save(target_dir + 'Images/' + row['image_id'] + '.jpg')


# --------これ以下はコメントアウト可能--------

#ランダムチェック
random_check = round(len(df_resized)/100)
print('オリジナルの長さ: ' + str(len(df)))
print('リサイズ後の長さ: ' + str(len(df_resized)))
print('1/100のランダムチェック: ' + str(random_check))

#画像の変換
for index, row in df_resized.sample(n=random_check).iterrows():
    #img_origin = Image.open(origin_dir + 'Images/' + row['image_id'] + '.jpg')
    img_resized = Image.open(target_dir + 'Images/' + row['image_id'] + '.jpg')
    draw = ImageDraw.Draw(img_resized)
    draw.rectangle(((row['xmin'], row['ymin']), (row['xmax'], row['ymax'])), fill=None, outline=(255, 255, 255))
    #plt.imshow(img_origin)
    #plt.show()
    plt.imshow(img_resized)
    plt.show()
