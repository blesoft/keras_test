import os
import cv2 as cv

DATA_DIR = "./pic_Food_sample/downloads/"
SAVE_DIR = "./pic_Food/"
categories = ["meat","noodles","seafood","sweets"]
IMG_SIZE = 50

for category in categories:
    path_data = os.path.join(DATA_DIR,category)
    path_save = os.path.join(SAVE_DIR,category)
    NUM = 0
    for image_name in os.listdir(path_data):
        # 画像読み込み
        img_array_origin = cv.imread(os.path.join(path_data,image_name))
        if not img_array_origin is None:
            height = img_array_origin.shape[0]
            weight = img_array_origin.shape[1]
            tan = height / weight
            if 0.5 < tan < 2.0:
                # print(tan)
                # リサイズ
                image_array_0 = cv.resize(img_array_origin,(IMG_SIZE,IMG_SIZE))
                # 上下左右反転（データ水増し）
                image_array_1 = cv.flip(image_array_0,0)
                image_array_2 = cv.flip(image_array_0,1)
                image_array_3 = cv.flip(image_array_0,-1)
                # 保存
                TITLE_0 = category + str(NUM) + '_0.jpg'
                TITLE_1 = category + str(NUM) + '_1.jpg'
                TITLE_2 = category + str(NUM) + '_2.jpg'
                TITLE_3 = category + str(NUM) + '_3.jpg'
                cv.imwrite(os.path.join(path_save,TITLE_0),image_array_0)
                cv.imwrite(os.path.join(path_save,TITLE_1),image_array_1)
                cv.imwrite(os.path.join(path_save,TITLE_2),image_array_2)
                cv.imwrite(os.path.join(path_save,TITLE_3),image_array_3)
                NUM += 1