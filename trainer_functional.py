from sys import argv
import tensorflow.keras.backend as K
from tensorflow import cast, float32
from tensorflow import keras
import numpy as np

def IOU_coef(y_pred, y_true, smooth=1):
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersect
    iou = K.mean((intersect + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])   
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def dice_coef_np(y_true, y_pred):
    return np.sum(y_pred[y_true == 1.0])*2.0/(np.sum(y_true) + np.sum(y_pred))

def down(x, filters, k_size=(3, 3), padding='same', strides=1):
    con = keras.layers.Conv2D(filters, k_size, padding=padding, strides=strides, activation='relu')(x)
    con = keras.layers.Conv2D(filters, k_size, padding=padding, strides=strides, activation='relu')(x)
    pool = keras.layers.MaxPool2D((2, 2), (2, 2))(con)
    return con, pool
    

def up(x, skip, filters, k_size=(3, 3), padding='same', strides=1):
    upsam = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([upsam, skip])
    con = keras.layers.Conv2D(filters, k_size, padding=padding, strides=strides, activation='relu')(concat)
    con = keras.layers.Conv2D(filters, k_size, padding=padding, strides=strides, activation='relu')(con)
    return con


def base(x, filters, k_size=(3, 3), padding='same', strides=1):
    con = keras.layers.Conv2D(filters, k_size, padding=padding, strides=strides, activation='relu')(x)
    con = keras.layers.Conv2D(filters, k_size, padding=padding, strides=strides, activation='relu')(con)
    return con


def Unet(img_size):
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((img_size, img_size, 3))
    
    pool0 = inputs
    con1, pool1 = down(pool0, f[0])
    con2, pool2 = down(pool1, f[1])
    con3, pool3 = down(pool2, f[2])
    con4, pool4 = down(pool3, f[3])
    
    b = base(pool4, f[4])
    
    up1 = up(b, con4, f[3])
    up2 = up(up1, con3, f[2])
    up3 = up(up2, con2, f[1])
    up4 = up(up3, con1, f[0])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up4)
    model = keras.models.Model(inputs, outputs)
    
    return model

