from trainer_functional import IOU_coef, Unet
from data_preprocessing import get_train, augmentation
from sklearn.model_selection import train_test_split
from numpy import concatenate
from tensorflow.keras.optimizers import Adam
from sys import argv
import os

path2train = "stage1_train/"

train_ids = next(os.walk(path2train))[1]

print(argv)
try:

	IMG_SIZE = int(argv[1])
	N_epochs = int(argv[2])
	B_size = int(argv[3])
	learning_rate = float(argv[4])

except:
	print("Not all params were given\nBreaking...")
	exit(0)

X, Y, origin_size = get_train(train_ids, IMG_SIZE, path2train)

print(X.max(), X.dtype)
augmented_x, augmented_y = augmentation(X, Y, 2, IMG_SIZE)
augmented_x = concatenate([augmented_x, X], axis=0)
augmented_y = concatenate([augmented_y, Y], axis=0)

train_x, test_x, train_y, test_y = train_test_split(augmented_x, augmented_y, test_size=0.2, random_state=1)

model = Unet(IMG_SIZE)
optim = Adam(lr=learning_rate)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[IOU_coef])

print("START FITTING...\nIMG_SIZE={}".format(IMG_SIZE))
model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=N_epochs, batch_size=B_size)

print("SAVING MODEL...")
model.save("my_model")

