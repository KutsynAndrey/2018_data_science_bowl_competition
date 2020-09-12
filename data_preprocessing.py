import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import imgaug.augmenters as iaa


def get_train(train_ids, img_size, path2train):
    
    X = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.float)
    Y = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
    origin_size = []
	
    print("LOADING TRAIN...")

    for i, ID in tqdm(enumerate(train_ids), total=len(train_ids), desc="TRAIN"):
        
        path = path2train + ID

        img = cv2.imread(path + "/images/" + ID + ".png")
        origin_size.append(img.shape)
        img = cv2.resize(img, (img_size, img_size)).astype("float")/255
        
        y = np.zeros((img_size, img_size, 1), dtype=np.bool)
        
        for mask_name in next(os.walk(path+"/masks/"))[2]:
            
            mask = cv2.imread(path + "/masks/" + mask_name)
            mask = cv2.resize(mask, (img_size, img_size))[:, :, 0]

            y = np.maximum(y, np.expand_dims(mask, axis=2))
        
        X[i] = img
        Y[i] = y
       
    return X, Y, origin_size
    

def get_test(stage1_ids, stage2_ids, img_size, path1, path2):
    
    stage1 = np.zeros((len(stage1_ids), img_size, img_size, 3), dtype=np.float)
    stage2 = np.zeros((len(stage2_ids), img_size, img_size, 3), dtype=np.float)
    origin_stage1 = []
    origin_stage2 = []

    print("LOADING STAGE1...")

    for i, ID in tqdm(enumerate(stage1_ids), total=len(stage1_ids), desc="STAGE1"):
        
        path = path1 + ID
        
        img = cv2.imread(path + "/images/" + ID + ".png")
        origin_stage1.append((img.shape[0], img.shape[1]))

        img = cv2.resize(img, (img_size, img_size)).astype("float")/255
        
        stage1[i] = img

    print("LOADING STAGE2...")
    for i, ID in tqdm(enumerate(stage2_ids), total=len(stage2_ids), desc="STAGE2"):
        
        path = path2 + ID
        
        img = cv2.imread(path + "/images/" + ID + ".png")
        origin_stage2.append((img.shape[0], img.shape[1]))

        img = cv2.resize(img, (img_size, img_size)).astype("float")/255
        
        stage2[i] = img


    return stage1, stage2, origin_stage1, origin_stage2

def get_original_train(train_ids, path2train):
    
    print("LOADING ORIGINAL...")

    original_img = []
    original_mask = []

    for i, ID in tqdm(enumerate(train_ids), total=len(train_ids), desc="TRAIN"):
        
        path = path2train + ID

        img = cv2.imread(path + "/images/" + ID + ".png")
        
        y = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        
        for mask_name in next(os.walk(path+"/masks/"))[2]:
            
            mask = cv2.imread(path + "/masks/" + mask_name)[:, :, 0]
            
            y = np.maximum(y, np.expand_dims(mask, axis=2))
        original_img.append(img)
        original_mask.append(y)

    return original_img, original_mask



def augmentation(X, Y, epochs, img_size):
    
    print("START AUGMENTATION...")

    augmented_x = np.zeros((X.shape[0]*epochs, img_size, img_size, 3), dtype=np.float)
    augmented_y = np.zeros((X.shape[0]*epochs, img_size, img_size, 1), dtype=np.bool)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.Rotate((-40, 40))
    ])

    for e in range(epochs):
        for i in range(X.shape[0]):
            img, seg = seq(images=[X[i]], segmentation_maps=[Y[i]])
            augmented_x[X.shape[0]* e + i] = img[0]
            augmented_y[X.shape[0]* e + i] = seg[0]
    
    print("AUGMENTED")            
    return augmented_x, augmented_y


def balanced_augmentation(X, Y, epochs, img_size):

    print("START AUGMENTATION...")

    cluster_df = pd.read_csv('cluster_df.csv', usecols=['cluster'])

    cluster_size = [0, 0, 0]

    cluster_size[0] = cluster_df.value_counts()[0].iloc[0]
    cluster_size[1] = cluster_df.value_counts()[1].iloc[0]
    cluster_size[2] = cluster_df.value_counts()[2].iloc[0]
    
    d_len = cluster_size[0]*epochs[0] + cluster_size[1]*epochs[1] + cluster_size[2]*epochs[2]

    augmented_x = np.zeros((d_len, img_size, img_size, 3), dtype=np.float)
    augmented_y = np.zeros((d_len, img_size, img_size, 1), dtype=np.bool)   


    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.Rotate((-40, 40))
    ])


    global_cnt=0

    for i in range(X.shape[0]):

        img_cluster = cluster_df.iloc[i]['cluster']

        '''if img_cluster == 2:
            print(epochs[img_cluster])
            cv2.imshow("img", X[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

        for e in range(epochs[img_cluster]):

            img, seg = seq(images=[X[i]], segmentation_maps=[Y[i]])
            augmented_x[global_cnt] = img[0]
            augmented_y[global_cnt] = seg[0]
            global_cnt+=1

    print("AUGMENTED")            

    return augmented_x, augmented_y    



def return_size(img, origin_i):
    img = cv2.resize(np.squeeze(img), (origin_i[1], origin_i[0]))
    return img
