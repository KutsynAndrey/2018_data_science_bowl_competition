import os
import sys
import cv2
from cv2 import resize
from tensorflow import keras
from data_preprocessing import get_test, return_size, get_original_train
from numpy import concatenate, random, expand_dims, squeeze
from trainer_functional import IOU_coef, dice_coef


img_size = None

try:
	img_size = int(sys.argv[1])
except:
	print("Image size was not given")
	exit(0)


path1 = "stage1_train/"

train_ids = next(os.walk(path1))[1]

X, Y = get_original_train(train_ids, path1)

model = keras.models.load_model("my_model", custom_objects={"IOU_coef": IOU_coef})

cv2.namedWindow("input")
cv2.namedWindow("predict")

while True:

	ix = random.randint(0, len(X))

	img = expand_dims(resize(X[ix].copy(), (img_size, img_size)), axis=0)

	predict = model.predict(x=img, batch_size=1)

	img = X[ix]
	predict = return_size(predict, img.shape)

	predict = (predict > 0.5).astype("float")

	cv2.imshow("input", squeeze(img))
	cv2.imshow("predict", squeeze(predict))

	key = cv2.waitKey(0)

	if key == ord('a'):
		break


cv2.destroyAllWindows()