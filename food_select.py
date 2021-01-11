from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

# 保存したモデルの読み込み
model = model_from_json(open('food_predict.json').read())
# 保存した重みの読み込み
model.load_weights('food_predict.hdf5')

categories = ["meat","noodles","seafood","sweets"]

# 画像を読み込む
IMG_PATH = str(input())
img = image.load_img(IMG_PATH,target_size=(50,50,3))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)

# 予測
features = model.predict(x)

# 予測結果
if features[0,0] == 1:
    print("お肉")
elif features[0,1] == 1:
    print("麺類")
elif features[0,2] == 1:
    print("海鮮")
else:
    print("お菓子")