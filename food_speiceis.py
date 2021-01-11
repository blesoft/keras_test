### モデル構築 ###
import matplotlib.pyplot as plt
from keras import layers,models
from keras.utils import np_utils
# from keras.callbacks import LearningRateScheduler
# from keras import optimizers
import dataset_pre_food


IMG_SIZE = 50
EPOCHS = 100
BATCH_SIZE = 256
CAT = len(dataset_pre_food.categories)
OPTIMIZERS = ["SGD","AdaGrad","RMSprop","AdaDelta","Adam"]
OPT = OPTIMIZERS[2] #RMSprop
x_train = dataset_pre_food.x_train.astype("float") / 255
y_train = dataset_pre_food.y_train
y_train = np_utils.to_categorical(y_train,CAT)
x_test  = dataset_pre_food.x_test.astype("float") / 255
y_test  = dataset_pre_food.y_test
y_test  = np_utils.to_categorical(y_test,CAT)
# # 学習率減衰
# def step_decay(epoch):
#     lr = 1e-3
#     if lr >= 50:
#         lr / 5.0
#     if lr >= 80:
#         lr / 2.0
#     return lr

# モデル構造　plttype
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32,(3,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,(3,3),padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64,(3,3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128,(3,3),padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128,(3,3)))
model.add(layers.Activation('relu'))
model.add(layers.GlobalAveragePooling2D())


model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(CAT))
model.add(layers.Activation('softmax'))

# # モデル構造 deeptype
# model = models.Sequential()
# model.add(layers.Conv2D(64,(3,3),padding='same',input_shape=x_train.shape[1:]))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(64,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Dropout(0.25))

# model.add(layers.Conv2D(128,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(128,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Dropout(0.25))

# model.add(layers.Conv2D(256,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(256,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(256,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(256,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(256,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(512,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.Conv2D(512,(3,3),padding='same'))
# model.add(layers.Activation('relu'))
# model.add(layers.GlobalAveragePooling2D())

# model.add(layers.Flatten())
# model.add(layers.Dense(1024))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1024))
# model.add(layers.Activation('relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(CAT))
# model.add(layers.Activation('softmax'))


# モデルのコンパイル
model.compile(loss="categorical_crossentropy",
              optimizer=OPT,
              metrics=["acc"])

# モデルの学習
# lr_decay = LearningRateScheduler(step_decay)
model = model.fit(x_train,y_train,
                  epochs=EPOCHS,batch_size=BATCH_SIZE,
                  validation_data=(x_test,y_test))#,callbacks=[lr_decay])

# 学習結果表示
acc = model.history["acc"]
val_acc = model.history["val_acc"]
loss = model.history["loss"]
val_loss = model.history["val_loss"]

epochs = range(len(acc))

plt.plot(epochs,acc,"bo",label="training acc")
plt.plot(epochs,val_acc,"b",label="validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
SAVE_ACC = "accuracy.jpg"
plt.savefig(SAVE_ACC)
plt.figure()

plt.plot(epochs,loss,"bo",label="trainig loss")
plt.plot(epochs,val_loss,"b",label="validation loss")
plt.title("Training and validation loss")
plt.legend()
SAVE_LOSS = "loss.jpg"
plt.savefig(SAVE_LOSS)

# モデルの保存
json_string = model.model.to_json()
open("food_predict.json","w").write(json_string)
# 重みの保存
HDF5_FILE = "food_predict.hdf5"
model.model.save_weights(HDF5_FILE)

# モデルの予測精度計測
score = model.model.evaluate(x_test,y_test)
print(score[1])

# # 各勾配法の精度比較
# score_all = []
# for opt in OPTIMIZERS:
#     # モデルのコンパイル
#     print(opt)
#     model.compile(loss="categorical_crossentropy",
#                   optimizer=opt,
#                   metrics=["acc"])
#     # モデルの学習
#     result = model.fit(x_train,y_train,
#                         epochs=EPOCHS,batch_size=BATCH_SIZE,
#                         validation_data=(x_test,y_test))
#     # 学習結果表示
#     acc = result.history["acc"]
#     val_acc = result.history["val_acc"]
#     loss = result.history["loss"]
#     val_loss = result.history["val_loss"]

#     epochs = range(len(acc))

#     plt.plot(epochs,acc,"bo",label="training acc")
#     plt.plot(epochs,val_acc,"b",label="validation acc")
#     plt.title("Training and Validation accuracy")
#     plt.legend()
#     SAVE_ACC = "accuracy_" + opt + ".jpg"
#     plt.savefig(SAVE_ACC)
#     plt.figure()

#     plt.plot(epochs,loss,"bo",label="trainig loss")
#     plt.plot(epochs,val_loss,"b",label="validation loss")
#     plt.title("Training and validation loss")
#     plt.legend()
#     SAVE_LOSS = "loss_" + opt + ".jpg"
#     plt.savefig(SAVE_LOSS)
#     plt.figure()

#     # # モデルの保存
#     # json_string = model.model.to_json()
#     # open("food_predict.json","w").write(json_string)
#     # # 重みの保存
#     # HDF5_FILE = "food_predict.hdf5"
#     # model.model.save_weights(HDF5_FILE)

#     # モデルの予測精度計測
#     score = result.model.evaluate(x_test,y_test)
#     score_all.append(score[1])
