import os
import cv2
from tensorflow import keras
import sys
from data_preprocessing import get_test, get_train, return_size, get_original_train
from numpy import concatenate, random, expand_dims, squeeze
from trainer_functional import IOU_coef, dice_coef_np
from tqdm import tqdm


img_size = None

try:
	img_size = int(sys.argv[1])
except:
	"Image size was not given"
	exit(0)

path1 = "stage1_test/"
path2 = "stage2_test_final/"
train_path = "stage1_train/"


stage1_ids = next(os.walk(path1))[1]
stage2_ids = next(os.walk(path2))[1]
train_ids = next(os.walk(train_path))[1]

#stage1, stage2, or_stage1, or_stage2 = get_test(stage1_ids, stage2_ids, img_size, path1, path2)
X, Y = get_original_train(train_ids, train_path)
model = keras.models.load_model("my_balanced_model", custom_objects={"IOU_coef": IOU_coef})

print("GENERATE DICE COEF...")

min_dice_score = 1e10
min_ID = None
for i, ID in tqdm(enumerate(train_ids), total=len(train_ids), desc="DICE COEF"):
        

        path =  train_path + ID

        try:
        	os.mkdir(path+"/predict")
        except FileExistsError:
        	pass

        img = cv2.imread(path + "/images/" + ID + ".png")
        img = cv2.resize(img, (img_size, img_size)).astype('float')/255

        predict = model.predict(x=expand_dims(img, axis=0), batch_size=1)
        predict = (predict > 0.5).astype('float')
        predict = return_size(squeeze(predict), X[i].shape)
        dice_score = dice_coef_np(squeeze(Y[i].astype('float'))/255, squeeze(predict))

        if dice_score < min_dice_score:
        	min_dice_score = dice_score
        	min_ID = ID

        with open(path + "/predict/dice_score.txt", "w") as f:
        	f.write("{}".format(dice_score))

        predict = expand_dims(predict, axis=2)
        predict = concatenate([predict, predict, predict], axis=2)*255
        cv2.imwrite(path + "/predict/" + ID + ".png", predict)

print("MIN DICE", min_dice_score)
print("ID", min_ID)

'''print("GENERATE STAGE1 MASKS...")



for i, ID in tqdm(enumerate(stage1_ids), total=len(stage1_ids), desc="STAGE1"):
        

        path = path1 + ID

        try:
        	os.mkdir(path+"/predict")
        except FileExistsError:
        	pass

        img = cv2.imread(path + "/images/" + ID + ".png")
        img = cv2.resize(img, (img_size, img_size)).astype('float')/255

        predict = model.predict(x=expand_dims(img, axis=0), batch_size=1)
        predict = (predict > 0.5).astype('float')
        predict = return_size(squeeze(predict), or_stage1[i])

        cv2.imwrite(path + "/predict/" + ID + ".png", squeeze(predict))


print("GENERATE STAGE2 MASKS...")

for i, ID in tqdm(enumerate(stage2_ids), total=len(stage2_ids), desc="STAGE2"):
        

        path = path2 + ID

        try:
        	os.mkdir(path+"/predict")
        except FileExistsError:
        	pass

        img = cv2.imread(path + "/images/" + ID + ".png")
        img = cv2.resize(img, (img_size, img_size)).astype('float')/255

        predict = model.predict(x=expand_dims(img, axis=0), batch_size=1)
        predict = return_size(squeeze(predict), or_stage2[i])
        predict = (predict > 0.5).astype('float')

        cv2.imwrite(path + "/predict/" + ID + ".png", squeeze(predict))'''