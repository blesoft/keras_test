import os
import cv2 as cv

data_dir = "./pic_Food/"
categories = ["don","egg","noodle","pasta","sarada","soup","sushi"]
IMG_SIZE = 50

for category in categories:
    path = os.path.join(data_dir,category)
    path2 = path + "_img"
    num = 0
    for image_name in os.listdir(path):
        print(image_name)
        # 画像読み込み
        img_array = cv.imread(os.path.join(path,image_name))
        # リサイズ
        image_resize = cv.resize(img_array,(IMG_SIZE,IMG_SIZE))
        # 保存
        title = category + str(num) + '.jpg'
        print(os.path.join(path2,title))
        cv.imwrite(os.path.join(path2,title),image_resize)
        num += 1