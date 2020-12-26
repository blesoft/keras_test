import os
import cv2 as cv

data_dir = "./pic_Food/"
categories = ["don_img","egg_img","noodle_img","pasta_img","sarada_img","soup_img","sushi_img"]
IMG_SIZE = 50

for category in categories:
    path = os.path.join(data_dir,category)
    num = 0
    for image_name in os.listdir(path):
        # 画像読み込み
        img_array = cv.imread(os.path.join(path,image_name))
        # 画像上下反転
        img_array_1 = cv.flip(img_array,0)
        img_array_2 = cv.flip(img_array,1)
        img_array_3 = cv.flip(img_array,-1)
        # 保存
        title_0 = category + str(num) + '_0' + '.jpg'
        title_1 = category + str(num) + '_1' + '.jpg'
        title_2 = category + str(num) + '_2' + '.jpg'
        title_3 = category + str(num) + '_3' + '.jpg'
        cv.imwrite(os.path.join(path,title_0),img_array)
        cv.imwrite(os.path.join(path,title_1),img_array_1)
        cv.imwrite(os.path.join(path,title_2),img_array_2)
        cv.imwrite(os.path.join(path,title_3),img_array_3)
        num += 1