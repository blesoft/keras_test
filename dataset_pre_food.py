import os
import random
import math

import cv2 as cv
import numpy as np

DATA_DIR = "./pic_Food/"
categories = ["meat","noodles","seafood","sweets"]
all_data = []

# import matplotlib.pyplot as plt
# ### 画像読み込みチェック ###
# for category in categories:
#     path = os.path.join(DATA_DIR,category)
#     for image_name in os.listdir(path):
#         img_array = cv.imread(os.path.join(path,image_name))
#         img_array = cv.cvtColor(img_array,cv.COLOR_BGR2RGB)
#         plt.imshow(img_array,cmap="gray")
#         plt.show()
#         break
#     break

def create_training_data():
    for class_num,category in enumerate(categories):
        path = os.path.join(DATA_DIR,category)
        for image_name in os.listdir(path):
            # 画像読み込み
            img_array = cv.imread(os.path.join(path,image_name))
            img_array = cv.cvtColor(img_array,cv.COLOR_BGR2RGB)
            # training_data にデータを追加
            all_data.append([img_array,class_num])

create_training_data()
random.shuffle(all_data) # データをシャッフル
th = math.floor(len(all_data)*0.8) # 8割りを tarining_data とする
training_data = all_data[0:th]
test_data     = all_data[th:]

# 画像データ
x_train = []
x_test = []
# ラベルデータ
y_train = []
y_test = []

# データセット作成
for feature , label in training_data:
    x_train.append(feature)
    y_train.append(label)
# # numpy 配列に変換
x_train = np.array(x_train)
y_train = np.array(y_train)

# テストデータ作成
for feature , label in test_data:
    x_test.append(feature)
    y_test.append(label)
# # numpy 配列に変換
x_test = np.array(x_test)
y_test = np.array(y_test)
# # データを保存する
# xy = (x_train,x_test,y_train,y_test)
# np.save("food_data.npy",xy)

# import matplotlib.pyplot as plt
# # データセット確認
# for i in range(0,4):
#     print("学習データのラベル：",y_train[i])
#     plt.subplot(2,2,i+1)
#     plt.axis("off")
#     if y_train[i] == 0:
#         label_name = "meat"
#     if y_train[i] == 1:
#         label_name = "noodles"
#     if y_train[i] == 2:
#         label_name = "seafood"
#     if y_train[i] == 3:
#         label_name = "sweets"
#     plt.title(label=label_name)
#     plt.imshow(x_train[i],cmap="gray")
# plt.show()