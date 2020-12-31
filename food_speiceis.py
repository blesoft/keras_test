### モデル構築 ###
import matplotlib.pyplot as plt
from keras import layers,models
from keras.utils import np_utils
from keras import optimizers
import dataset_pre_food

IMG_SIZE = 75
CAT = len(dataset_pre_food.categories)
x_train = dataset_pre_food.x_train.astype("float") / 255
y_train = dataset_pre_food.y_train
y_train = np_utils.to_categorical(y_train,CAT)
x_test  = dataset_pre_food.x_test.astype("float") / 255
y_test  = dataset_pre_food.y_test
y_test  = np_utils.to_categorical(y_test,CAT)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(CAT,activation="softmax")) #分類先の種類分設定

# モデルのコンパイル
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])

# モデルの学習
model = model.fit(x_train,y_train,
                  epochs=15,batch_size=32,
                  validation_data=(x_test,y_test))

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
plt.savefig("accuacy.jpg")
plt.figure()

plt.plot(epochs,loss,"bo",label="trainig loss")
plt.plot(epochs,val_loss,"b",label="validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("loss.jpg")

# # モデルの保存
# json_string = model.model.to_json()
# open("food_predict.json","w").write(json_string)
# # 重みの保存
# hdf5_file = "food_predict.hdf5"
# model.model.save_weights(hdf5_file)

# モデルの予測精度計測
score = model.model.evaluate(x_test,y_test)
print(score[1])