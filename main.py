# import glob
# import random
# import math
# import numpy as np
# from PIL import Image
# from keras import layers,models,optimizers
# from keras.utils import np_utils

# ### 学習/検証データの作成 ###
# # 画像が保存されているディレクトリのパス
# path_dir = "pic_Food"
# # 画像がそれぞれ保存されているフォルダ名
# categories = ["スープ","パスタ","ラーメン","寿司","丼"]

# # 各データ
# ## 画像
# X = []
# ## ラベル
# Y = []
# ## ラベルと画像を紐づけしたデータ
# allfiles = []

# # allfiles リストの作成
# for idx , cat in enumerate(categories):
#     image_path = path_dir + "/" + cat
#     files = glob.glob(image_path + "/*.jpg")
#     for f in files:
#         allfiles.append((idx,f))

# # シャッフルした後、学習データと検証データを作成する
# random.shuffle(allfiles)
# th = math.floor(len(allfiles) * 0.8)
# train = allfiles[0:th] # 8割りを学習データ
# test = allfiles[th:] # 残り2割りを検証データ
# ## list作成関数
# def make_sample(file):
#     x = []
#     y = []
#     for i_cat,fname in file:
#         img = Image.open(fname)
#         img = img.convert("RGB")
#         img = img.resize((150,150))
#         data = np.asarray(img)
#         x.append(data)
#         y.append(i_cat)
#     return np.array(x),np.array(y)
# x_train , x_test = make_sample(train)
# y_train , y_test = make_sample(test)
# xy = (x_train,x_test,y_train,y_test)
# np.save("food_data.npy",xy) # 拡張子npyで保存

# ### モデルの構築 ###
# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128,(3,3),activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512,activation="relu"))
# model.add(layers.Dense(10,activation="sigmoid"))
# # model.summary()
# ## モデルコンパイル
# model.compile(loss="binary_crossentropy",
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=["acc"])
# ## データの正規化
# nb_classes = len(categories)
# x_train = x_train.astype("float") / 255
# x_test = x_test.astype("float") / 255
# # print(y_train)
# ## keras で扱えるように categories をベクトルに変換
# y_train = np_utils.to_categorical(y_train,nb_classes)
# y_test = np_utils.to_categorical(y_test,nb_classes)
# # ## モデルの学習
# # model = model.fit(x_train,y_train,epochs=10,batch_size=6,validation_data=(x_test,y_test))