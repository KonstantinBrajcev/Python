import check_for_errors
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model, Sequential # РџРѕРґРєР»СЋС‡Р°РµРј РјРѕРґРµР»СЊ С‚РёРїР° Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.utils import to_categorical, plot_model # РџРѕР»РєСЋС‡Р°РµРј РјРµС‚РѕРґС‹ .to_categorical() Рё .plot_model()
sns.set_style('darkgrid') 
from tensorflow.keras import backend as K # РРјРїРѕСЂС‚РёСЂСѓРµРј РјРѕРґСѓР»СЊ backend keras'Р°
from tensorflow.keras.optimizers import Nadam, RMSprop, Adadelta,Adam # РРјРїРѕСЂС‚РёСЂСѓРµРј РѕРїС‚РёРјРёР·Р°С‚РѕСЂ Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from tensorflow.keras.models import Model # РРјРїРѕСЂС‚РёСЂСѓРµРј РјРѕРґРµР»Рё keras: Model
from tensorflow.keras.layers import Input, RepeatVector, Conv2DTranspose, concatenate, Activation, Embedding, Input, MaxPooling2D, Conv2D, BatchNormalization # РРјРїРѕСЂС‚РёСЂСѓРµРј СЃС‚Р°РЅРґР°СЂС‚РЅС‹Рµ СЃР»РѕРё keras
import importlib.util, sys, gdown,os
import tensorflow as tf
from PIL import Image
import pandas as pd
import time
from IPython.display import clear_output
from tensorflow.keras.preprocessing import image
import termcolor
from termcolor import colored
from google.colab import files
import subprocess, os, warnings, time
from pandas.errors import SettingWithCopyWarning
from subprocess import STDOUT, check_call
from IPython import display
import numpy as np
import requests
import random, pickle
import zipfile
from sklearn.model_selection import train_test_split
import ast
import json
from tabulate import tabulate
import getpass
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings('ignore')
import logging
tf.get_logger().setLevel(logging.ERROR)

@check_for_errors.decorator_check
def СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(РґР°РЅРЅС‹Рµ, **kwargs):
  args={}
  if 'РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ' in kwargs:
    args['input_shape'] = kwargs['РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ']  
  РїР°СЂР°РјРµС‚СЂС‹ = [РґР°РЅРЅС‹Рµ]
  act = 'relu'
  if '-' in РґР°РЅРЅС‹Рµ:
    РїР°СЂР°РјРµС‚СЂС‹ = РґР°РЅРЅС‹Рµ.split('-')
  if РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РџРћР›РќРћРЎР’РЇР—РќР«Р™':    
    if len(РїР°СЂР°РјРµС‚СЂС‹)>2:
      act = РїР°СЂР°РјРµС‚СЂС‹[2]
    return Dense(int(РїР°СЂР°РјРµС‚СЂС‹[1]), activation=act, **args)
  if РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РџРћР’РўРћР ':
    return RepeatVector(int(РїР°СЂР°РјРµС‚СЂС‹[1]))
  if РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'Р­РњР‘Р•Р”Р”РРќР“':
    return Embedding(int(РїР°СЂР°РјРµС‚СЂС‹[2]), int(РїР°СЂР°РјРµС‚СЂС‹[1]), input_length=int(РїР°СЂР°РјРµС‚СЂС‹[3]))
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РЎР’Р•Р РўРћР§РќР«Р™2D':
    if len(РїР°СЂР°РјРµС‚СЂС‹)<5:
      act = 'relu'
      pad='same'
    else:
      act = РїР°СЂР°РјРµС‚СЂС‹[4]
      pad = РїР°СЂР°РјРµС‚СЂС‹[3]
    if any(i in '()' for i in РїР°СЂР°РјРµС‚СЂС‹[2]):
      return Conv2D(int(РїР°СЂР°РјРµС‚СЂС‹[1]), (int(РїР°СЂР°РјРµС‚СЂС‹[2][1]),int(РїР°СЂР°РјРµС‚СЂС‹[2][3])), padding=pad,activation=act, **args)
    else:
      return Conv2D(int(РїР°СЂР°РјРµС‚СЂС‹[1]), int(РїР°СЂР°РјРµС‚СЂС‹[2]), padding=pad,activation=act, **args)
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РЎР’Р•Р РўРћР§РќР«Р™1D':
    if len(РїР°СЂР°РјРµС‚СЂС‹)>4:
      act = РїР°СЂР°РјРµС‚СЂС‹[4]
      pad = РїР°СЂР°РјРµС‚СЂС‹[3]
    else:
      act = 'relu'
      pad = 'same'
    return Conv1D(int(РїР°СЂР°РјРµС‚СЂС‹[1]), int(РїР°СЂР°РјРµС‚СЂС‹[2]), padding=pad,activation=act, **args)

  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'Р’Р«Р РђР’РќРР’РђР®Р©РР™':
    if 'input_shape' in args:
      return Flatten(input_shape=args['input_shape'])
    else:
      return Flatten() 
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РќРћР РњРђР›РР—РђР¦РРЇ':
    return BatchNormalization()
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РќРћР РњРђР›РР—РђР¦РРЇ_01':
    return Lambda(normalize_01)
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РќРћР РњРђР›РР—РђР¦РРЇ_11':
    return Lambda(normalize_m11)
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'Р”Р•РќРћР РњРђР›РР—РђР¦РРЇ':
    return Lambda(denormalize_m11)
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РђРљРўРР’РђР¦РРЇ':
    return Activation(РїР°СЂР°РјРµС‚СЂС‹[1])
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'Р›РЎРўРњ':    
    return LSTM(int(РїР°СЂР°РјРµС‚СЂС‹[1]), return_sequences=РїР°СЂР°РјРµС‚СЂС‹[2]=='РџРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅРѕСЃС‚СЊ', **args)
  
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РњРђРљРЎРџРЈР›Р›РРќР“2D':
    if any(i in '()' for i in РїР°СЂР°РјРµС‚СЂС‹[1]):
      return MaxPooling2D((int(РїР°СЂР°РјРµС‚СЂС‹[1][1]),int(РїР°СЂР°РјРµС‚СЂС‹[1][3])))
    else:
      return MaxPooling2D(int(РїР°СЂР°РјРµС‚СЂС‹[1]))  
  
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'РњРђРљРЎРџРЈР›Р›РРќР“1D':
    if any(i in '()' for i in РїР°СЂР°РјРµС‚СЂС‹[1]):
      return MaxPooling1D(int(РїР°СЂР°РјРµС‚СЂС‹[1]))
    else:
      return MaxPooling1D(int(РїР°СЂР°РјРµС‚СЂС‹[1]))
      
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'Р”Р РћРџРђРЈРў':
    return Dropout(float(РїР°СЂР°РјРµС‚СЂС‹[1]))
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'PRELU':
    return PReLU(shared_axes=[1, 2])
  elif РїР°СЂР°РјРµС‚СЂС‹[0].upper() == 'LEAKYRELU':
    return LeakyReLU(alpha=float(РїР°СЂР°РјРµС‚СЂС‹[1]))
  else:
    assert False, f'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: \'{РїР°СЂР°РјРµС‚СЂС‹[0]}\'.\
     Р’РѕР·РјРѕР¶РЅРѕ РІС‹ РёРјРµР»Рё РІРІРёРґСѓ \'{check_for_errors.check_word(РїР°СЂР°РјРµС‚СЂС‹[0], "layer")}\'.'

def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)

def normalize_01(x):
    return x / 255.0

# РќРѕСЂРјР°Р»РёР·СѓРµС‚ RGB РёР·РѕР±СЂР°Р¶РµРЅРёСЏ Рє РїСЂРѕРјРµР¶СѓС‚РєСѓ [-1, 1]
def normalize_m11(x):
    return x / 127.5 - 1

# РћР±СЂР°С‚РЅР°СЏ РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ
def denormalize_m11(x):
    return (x + 1) * 127.5

#РњРµС‚СЂРёРєР°
def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_РґР»СЏ_РєР»Р°СЃСЃРёС„РёРєР°С†РёРё(СЃР»РѕРё, РІС…РѕРґ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё):
  layers = СЃР»РѕРё.split()
  model = Sequential()
  layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0], РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ=РІС…РѕРґ)
  model.add(layer)   
  for i in range(1, len(layers)):
    layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
    model.add(layer)
  #print('РЎРѕР·РґР°РЅР° РјРѕРґРµР»СЊ РЅРµР№СЂРѕРЅРЅРѕР№ СЃРµС‚Рё!')
  РїР°СЂР°РјРµС‚СЂС‹ = РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё.split()  
  loss = РїР°СЂР°РјРµС‚СЂС‹[0].split(':')[1]
  opt = РїР°СЂР°РјРµС‚СЂС‹[1].split(':')[1]
  metrica = ''
  if (len(РїР°СЂР°РјРµС‚СЂС‹)>2):
    metrica = РїР°СЂР°РјРµС‚СЂС‹[2].split(':')[1]
  if metrica=='':
    model.compile(loss=loss, optimizer = opt)
  else:
    model.compile(loss=loss, optimizer = opt, metrics=[metrica])
  return model

def СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_РґР»СЏ_СЃРµРіРјРµРЅС‚Р°С†РёРё(СЃР»РѕРё, РІС…РѕРґ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё):
  def С‚РѕС‡РЅРѕСЃС‚СЊ(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
  layers = СЃР»РѕРё.split()
  model = Sequential()
  layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0], РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ=РІС…РѕРґ)
  model.add(layer) 
  for i in range(1, len(layers)):
    layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
    model.add(layer)
  #print('РЎРѕР·РґР°РЅР° РјРѕРґРµР»СЊ РЅРµР№СЂРѕРЅРЅРѕР№ СЃРµС‚Рё!')
  РїР°СЂР°РјРµС‚СЂС‹ = РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё.split()  
  loss = РїР°СЂР°РјРµС‚СЂС‹[0].split(':')[1]
  opt = РїР°СЂР°РјРµС‚СЂС‹[1].split(':')[1]
  metrica = ''  
  if (len(РїР°СЂР°РјРµС‚СЂС‹)>2):
    if РїР°СЂР°РјРµС‚СЂС‹[2].split(':')[1] == 'dice_coef':
      metrica = С‚РѕС‡РЅРѕСЃС‚СЊ
  model.compile(loss=loss, optimizer = opt, metrics =[metrica])
  return model

def СЃРѕР·РґР°С‚СЊ_РґРёСЃРєСЂРёРјРёРЅР°С‚РѕСЂ_РїРѕРІС‹С€РµРЅРёСЏ_СЂР°Р·РјРµСЂРЅРѕСЃС‚Рё(Р±Р»РѕРє_РґРёСЃРєСЂРёРјРёРЅР°С‚РѕСЂР°,РєРѕР»РёС‡РµСЃС‚РІРѕ_Р±Р»РѕРєРѕРІ, С„РёРЅР°Р»СЊРЅС‹Р№_Р±Р»РѕРє):
  x_in = Input(shape=(96, 96, 3))
  x = Lambda(normalize_m11)(x_in)
  blocks = Р±Р»РѕРє_РґРёСЃРєСЂРёРјРёРЅР°С‚РѕСЂР°.split()
  for i in range(РєРѕР»РёС‡РµСЃС‚РІРѕ_Р±Р»РѕРєРѕРІ):
    for b in blocks:
      x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(b) (x)      
  x = Flatten() (x)
  blocks = С„РёРЅР°Р»СЊРЅС‹Р№_Р±Р»РѕРє.split()
  for b in blocks:
    x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(b) (x)  
  return Model(x_in, x)

def СЃРѕР·РґР°С‚СЊ_РіРµРЅРµСЂР°С‚РѕСЂ_РїРѕРІС‹С€РµРЅРёСЏ_СЂР°Р·РјРµСЂРЅРѕСЃС‚Рё(СЃС‚Р°СЂС‚РѕРІС‹Р№_Р±Р»РѕРє, РѕСЃРЅРѕРІРЅРѕР№_Р±Р»РѕРє, С„РёРЅР°Р»СЊРЅС‹Р№_Р±Р»РѕРє):
  x_in = Input(shape=(None, None, 3))
  layers = СЃС‚Р°СЂС‚РѕРІС‹Р№_Р±Р»РѕРє.split()
  x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (x_in)
  for i in range(1, len(layers)):
    x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x)
  layers = РѕСЃРЅРѕРІРЅРѕР№_Р±Р»РѕРє.split()
  x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (x)
  for i in range(1, len(layers)-1):
    x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x2)
  x2 = Add()([x2, x2]) 
  x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[-1]) (x2)
  x = Add()([x, x2])
  x = upsample(x, 64 * 4)
  x = upsample(x, 64 * 4)
  layers = С„РёРЅР°Р»СЊРЅС‹Р№_Р±Р»РѕРє.split()
  x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (x)
  for i in range(1, len(layers)):
    x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x)
  return Model(x_in, x)

def СЃРѕР·РґР°С‚СЊ_UNET(**kwargs):
  РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё = 'РћС€РёР±РєР°:categorical_crossentropy\
      РѕРїС‚РёРјРёР·Р°С‚РѕСЂ:adam\
      РјРµС‚СЂРёРєР°:dice_coef'
  def isConv2D(layer):
    return layer.split('-')[0]=='РЎРІРµСЂС‚РѕС‡РЅС‹Р№2D'
      
  def С‚РѕС‡РЅРѕСЃС‚СЊ(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

  if 'РєРѕР»РёС‡РµСЃС‚РІРѕ_РІС‹С…РѕРґРЅС‹С…_РєР»Р°СЃСЃРѕРІ' in kwargs:
    n_classes = kwargs['РєРѕР»РёС‡РµСЃС‚РІРѕ_РІС‹С…РѕРґРЅС‹С…_РєР»Р°СЃСЃРѕРІ']
  else:
    n_classes = 2

  #Р‘Р»РѕРє РІРЅРёР· 
  block1 = []
  for i in range(len(kwargs['Р±Р»РѕРєРё_РІРЅРёР·'])):
    temp_block = kwargs['Р±Р»РѕРєРё_РІРЅРёР·'][i]
    split_block = temp_block.split()
    block1.append(split_block)
  
  #Р±Р»РѕРє РІРЅРёР·Сѓ
  s = kwargs['Р±Р»РѕРє_РІРЅРёР·Сѓ'].split('\n')
  down_blocks = ''
  for i in range(len(s)):
    temp_itm = s[i].split()
    for j in range(len(temp_itm)):
      if j + 1 == len(temp_itm) and i + 1 == len(s):
        down_blocks += str(temp_itm[j])
      else:
        down_blocks += str(temp_itm[j] + ' ')
  down_blocks = down_blocks.split() 
  
  #Р‘Р»РѕРє РІРІРµСЂС… 
  block2 = []
  for i in range(len(kwargs['Р±Р»РѕРєРё_РІРІРµСЂС…'])):
    temp_block = kwargs['Р±Р»РѕРєРё_РІРІРµСЂС…'][i]
    split_block = temp_block.split()
    block2.append(split_block)

  img_input = Input(kwargs['РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ']) #РІС…РѕРґРЅРѕР№ СЂР°Р·РјРµСЂ
  nBlock = len(kwargs['Р±Р»РѕРєРё_РІРІРµСЂС…']) + 1 #РєРѕР»РёС‡РµСЃС‚РІРѕ Р±Р»РѕРєРѕРІ
  С‚РµРєСѓС‰РёР№_Р±Р»РѕРє = 0
  b_o = []
  #Down
  #input layers
  layer = block1[0][0]
  if isConv2D(layer):
    layer = layer
  x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layer, РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ=kwargs['РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ'])(img_input)

  for i in range(len(block1)):
    for j in range(len(block1[i])):
      layer = block1[i][j]
      if isConv2D(layer):
        layer = layer
      x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layer)(x)
    b_o.append(x)
    x = MaxPooling2D()(b_o[-1])

  #Down block
  for i in range(nBlock-1):
    for j in range(len(down_blocks)):
      layer = down_blocks[j]
      if isConv2D(layer):
        layer = layer
      x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layer)(x)
    b_o.append(x)
    x = MaxPooling2D()(b_o[-1])
  x = b_o[i+1]   

  #Up
  for i in range(len(block2)):
    x = Conv2DTranspose(2**(2*nBlock-i), (2, 2), strides=(2, 2), padding='same')(x)    # Р”РѕР±Р°РІР»СЏРµРј СЃР»РѕР№ Conv2DTranspose СЃ 256 РЅРµР№СЂРѕРЅР°РјРё
    for j in range(len(block2[i])):
      layer = block2[i][j]
      if layer=='РћР±СЉРµРґРёРЅРµРЅРёРµ':
        x = concatenate([x, b_o[nBlock-i-2]])
      else:
        if isConv2D(layer):
          layer = layer    
        x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layer) (x)
  x = Conv2D(n_classes, (3, 3), activation='softmax', padding='same')(x)  # Р”РѕР±Р°РІР»СЏРµРј Conv2D-РЎР»РѕР№ СЃ softmax-Р°РєС‚РёРІР°С†РёРµР№ РЅР° num_classes-РЅРµР№СЂРѕРЅРѕРІ

  РјРѕРґ = Model(img_input, x)
  РїР°СЂР°РјРµС‚СЂС‹ = РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё.split()  
  loss = РїР°СЂР°РјРµС‚СЂС‹[0].split(':')[1]
  opt = РїР°СЂР°РјРµС‚СЂС‹[1].split(':')[1]
  metrica = ''  
  if (len(РїР°СЂР°РјРµС‚СЂС‹)>2):
    if РїР°СЂР°РјРµС‚СЂС‹[2].split(':')[1] == 'dice_coef':
      metrica = С‚РѕС‡РЅРѕСЃС‚СЊ
  
  #print('РЎРѕР·РґР°РЅР° РјРѕРґРµР»СЊ РЅРµР№СЂРѕРЅРЅРѕР№ СЃРµС‚Рё!')
  РјРѕРґ.compile(loss=loss, optimizer = opt, metrics =[metrica])
  return РјРѕРґ
  
def СЃРѕР·РґР°С‚СЊ_PSP(**kwargs):
  РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё = 'РћС€РёР±РєР°:categorical_crossentropy\
    РѕРїС‚РёРјРёР·Р°С‚РѕСЂ:adam\
    РјРµС‚СЂРёРєР°:dice_coef'
  def С‚РѕС‡РЅРѕСЃС‚СЊ(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
    
  img_input = Input(kwargs['РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ'])
  nBlock = kwargs['РєРѕР»РёС‡РµСЃС‚РІРѕ_Р±Р»РѕРєРѕРІ']
  if 'РєРѕР»РёС‡РµСЃС‚РІРѕ_РІС‹С…РѕРґРЅС‹С…_РєР»Р°СЃСЃРѕРІ' in kwargs:
    n_classes = kwargs['РєРѕР»РёС‡РµСЃС‚РІРѕ_РІС‹С…РѕРґРЅС‹С…_РєР»Р°СЃСЃРѕРІ']
  else:
    n_classes = 2
  start_block = kwargs['СЃС‚Р°СЂС‚РѕРІС‹Р№_Р±Р»РѕРє']
  layers = start_block.split()
  layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0])  
  x = layer(img_input)
  for i in range(1, len(layers)):
    layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
    x = layer(x)
  
  x_mp = []
  conv_size = 32
  block_PSP = kwargs['Р±Р»РѕРє_PSP']
  layers = block_PSP.split()  
  for i in range(nBlock):
    l = MaxPooling2D(2**(i+1))(x)
    for k in range(len(layers)):
      layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[k])      
      l = layer(l)
    l = Conv2DTranspose(32, (2**(i+1), 2**(i+1)), strides=(2**(i+1), 2**(i+1)), activation='relu')(l)
    x_mp.append(l)

  fin = concatenate([img_input]+ x_mp)

  final_block = kwargs['С„РёРЅР°Р»СЊРЅС‹Р№_Р±Р»РѕРє']+'-same-softmax'
  layers = final_block.split()
  for i in range(len(layers)):
    layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
    fin = layer(fin)

  РїР°СЂР°РјРµС‚СЂС‹ = РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё.split()  
  loss = РїР°СЂР°РјРµС‚СЂС‹[0].split(':')[1]
  opt = РїР°СЂР°РјРµС‚СЂС‹[1].split(':')[1]
  metrica = ''  
  if (len(РїР°СЂР°РјРµС‚СЂС‹)>2):
    if РїР°СЂР°РјРµС‚СЂС‹[2].split(':')[1] == 'dice_coef':
      metrica = С‚РѕС‡РЅРѕСЃС‚СЊ
  
  РјРѕРґ = Model(img_input, fin)
  #print('РЎРѕР·РґР°РЅР° РјРѕРґРµР»СЊ РЅРµР№СЂРѕРЅРЅРѕР№ СЃРµС‚Рё!')
  РјРѕРґ.compile(loss='categorical_crossentropy', optimizer=Adam(lr=3e-4), metrics =[С‚РѕС‡РЅРѕСЃС‚СЊ])
  return РјРѕРґ
  
def СЃРѕР·РґР°С‚СЊ_СЃРѕСЃС‚Р°РІРЅСѓСЋ_СЃРµС‚СЊ_РєРІР°СЂС‚РёСЂС‹(РґР°РЅРЅС‹Рµ, *РЅРµР№СЂРѕРЅРєРё):    
    input1 = Input(РґР°РЅРЅС‹Рµ[0].shape[1],)
    input2 = Input(РґР°РЅРЅС‹Рµ[1].shape[1],)
    
    layers = РЅРµР№СЂРѕРЅРєРё[0].split()
    x1 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (input1)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x1 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x1)

    layers = РЅРµР№СЂРѕРЅРєРё[1].split()
    x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (input2)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x2)
           
    x = concatenate([x1, x2])
    layers = РЅРµР№СЂРѕРЅРєРё[2].split()
    x3 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (x)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!='0', 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x3 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x3)    
    model = Model([input1, input2], x3)
    model.compile(loss="mae", optimizer=Nadam(lr=1e-3), metrics=["mae"])
    return model

def СЃРѕР·РґР°С‚СЊ_СЃРѕСЃС‚Р°РІРЅСѓСЋ_СЃРµС‚СЊ(РґР°РЅРЅС‹Рµ, РјРµС‚РєРё, *РЅРµР№СЂРѕРЅРєРё):
    img_input1 = Input(РґР°РЅРЅС‹Рµ[0].shape[1],)
    img_input2 = Input(РґР°РЅРЅС‹Рµ[1].shape[1],)
    img_input3 = Input(РґР°РЅРЅС‹Рµ[2].shape[1],)
    
    layers = РЅРµР№СЂРѕРЅРєРё[0].split()
    x1 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (img_input1)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x1 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x1)

    layers = РЅРµР№СЂРѕРЅРєРё[1].split()
    x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (img_input2)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x2)
    
    layers = РЅРµР№СЂРѕРЅРєРё[2].split()
    x3 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (img_input3)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x3 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x3)
        
    x = concatenate([x1, x2, x3])
    x = Dense(100, activation="relu")(x)
    x = Dense(РјРµС‚РєРё.shape[1], activation="softmax")(x)
    
    model = Model([img_input1, img_input2, img_input3], x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=5e-5), metrics=["accuracy"])
    return model

def СЃРѕР·РґР°С‚СЊ_СЃРѕСЃС‚Р°РІРЅСѓСЋ_СЃРµС‚СЊ_РїРёСЃР°С‚РµР»Рё(РґР°РЅРЅС‹Рµ, *РЅРµР№СЂРѕРЅРєРё):
    img_input1 = Input(РґР°РЅРЅС‹Рµ[0].shape[1],)
    img_input2 = Input(РґР°РЅРЅС‹Рµ[1].shape[1],)
    img_input3 = Input(РґР°РЅРЅС‹Рµ[2].shape[1],)
    
    layers = РЅРµР№СЂРѕРЅРєРё[0].split()
    x1 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (img_input1)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x1 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x1)

    layers = РЅРµР№СЂРѕРЅРєРё[1].split()
    x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (img_input2)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x2 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x2)
    
    layers = РЅРµР№СЂРѕРЅРєРё[2].split()
    x3 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[0]) (img_input3)
    for i in range(1, len(layers)):
        layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
        assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
        x3 = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x3)
        
    x = concatenate([x1, x2, x3])
    x = Dense(1024, activation="relu")(x)
    x = Dense(6, activation="softmax")(x)
    
    model = Model([img_input1, img_input2, img_input3], x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=5e-5), metrics=["accuracy"])
    return model

def СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_С‡Р°С‚_Р±РѕС‚(СЂР°Р·РјРµСЂ_СЃР»РѕРІР°СЂСЏ, СЌРЅРєРѕРґРµСЂ, РґРµРєРѕРґРµСЂ):
  encoderInputs = Input(shape=(None , )) # СЂР°Р·РјРµСЂС‹ РЅР° РІС…РѕРґРµ СЃРµС‚РєРё (Р·РґРµСЃСЊ Р±СѓРґРµС‚ encoderForInput)
  
  layers = СЌРЅРєРѕРґРµСЂ.split()
  if '-' in layers[0]:
    Р±СѓРєРІР°, РїР°СЂР°РјРµС‚СЂ = layers[0].split('-')
    x = Embedding(СЂР°Р·РјРµСЂ_СЃР»РѕРІР°СЂСЏ, int(РїР°СЂР°РјРµС‚СЂ), mask_zero=True) (encoderInputs)      
  for i in range(1, len(layers)-1):
    layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
    assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
    x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x)
  if '-' in layers[-1]:
    Р±СѓРєРІР°, РїР°СЂР°РјРµС‚СЂ = layers[-1].split('-')
  encoderOutputs, state_h , state_c = LSTM(int(РїР°СЂР°РјРµС‚СЂ), return_state=True)(x)
  encoderStates = [state_h, state_c]
    
  decoderInputs = Input(shape=(None, )) # СЂР°Р·РјРµСЂС‹ РЅР° РІС…РѕРґРµ СЃРµС‚РєРё (Р·РґРµСЃСЊ Р±СѓРґРµС‚ decoderForInput)
  layers = РґРµРєРѕРґРµСЂ.split()
  if '-' in layers[0]:
    Р±СѓРєРІР°, РїР°СЂР°РјРµС‚СЂ = layers[0].split('-')
    x = Embedding(СЂР°Р·РјРµСЂ_СЃР»РѕРІР°СЂСЏ, int(РїР°СЂР°РјРµС‚СЂ), mask_zero=True) (decoderInputs) 
  for i in range(1, len(layers)-1):
    layer = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i])
    assert layer!=0, 'РќРµРІРѕР·РјРѕР¶РЅРѕ РґРѕР±Р°РІРёС‚СЊ СѓРєР°Р·Р°РЅРЅС‹Р№ СЃР»РѕР№: '+layer
    x = СЃРѕР·РґР°С‚СЊ_СЃР»РѕР№(layers[i]) (x)
  if '-' in layers[-1]:
    Р±СѓРєРІР°, РїР°СЂР°РјРµС‚СЂ = layers[-1].split('-')
  decoderLSTM = LSTM(int(РїР°СЂР°РјРµС‚СЂ), return_state=True, return_sequences=True)
  decoderOutputs , _ , _ = decoderLSTM (x, initial_state=encoderStates)
  # Р РѕС‚ LSTM'Р° СЃРёРіРЅР°Р» decoderOutputs РїСЂРѕРїСѓСЃРєР°РµРј С‡РµСЂРµР· РїРѕР»РЅРѕСЃРІСЏР·РЅС‹Р№ СЃР»РѕР№ СЃ СЃРѕС„С‚РјР°РєСЃРѕРј РЅР° РІС‹С…РѕРґРµ
  decoderDense = Dense(СЂР°Р·РјРµСЂ_СЃР»РѕРІР°СЂСЏ, activation='softmax') 
  output = decoderDense (decoderOutputs)
  ######################
  # РЎРѕР±РёСЂР°РµРј С‚СЂРµРЅРёСЂРѕРІРѕС‡РЅСѓСЋ РјРѕРґРµР»СЊ РЅРµР№СЂРѕСЃРµС‚Рё
  ######################
  model = Model([encoderInputs, decoderInputs], output)
  model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
  return model        
    
def СЃС…РµРјР°_РјРѕРґРµР»Рё(РјРѕРґРµР»СЊ):
  print('РЎС…РµРјР° РјРѕРґРµР»Рё:')
  return plot_model(РјРѕРґРµР»СЊ, dpi=60) # Р’С‹РІРѕРґРёРј СЃС…РµРјСѓ РјРѕРґРµР»Рё

def СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ(СЃР»РѕРё, РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё, Р·Р°РґР°С‡Р°):
  layers = СЃР»РѕРё.split()
  Р·Р°РґР°С‡Р° = Р·Р°РґР°С‡Р°.lower()
  if Р·Р°РґР°С‡Р° == 'РєР»Р°СЃСЃРёС„РёРєР°С†РёСЏ РёР·РѕР±СЂР°Р¶РµРЅРёР№' or Р·Р°РґР°С‡Р° == 'РєР»Р°СЃСЃРёС„РёРєР°С†РёСЏ РІР°РєР°РЅСЃРёР№':
    РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё = 'РћС€РёР±РєР°:sparse_categorical_crossentropy\
            РѕРїС‚РёРјРёР·Р°С‚РѕСЂ:adam\
            РјРµС‚СЂРёРєР°:accuracy'
    return СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_РґР»СЏ_РєР»Р°СЃСЃРёС„РёРєР°С†РёРё(СЃР»РѕРё+'-softmax', РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё)
  if Р·Р°РґР°С‡Р° == 'РІСЂРµРјРµРЅРЅРѕР№ СЂСЏРґ':
    # РЈРєР°Р·С‹РІР°РµРј РїР°СЂР°РјРµС‚СЂС‹ СЃРѕР·РґР°РІР°РµРјРѕР№ РјРѕРґРµР»Рё
    РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё = 'РћС€РёР±РєР°:mse\
    РѕРїС‚РёРјРёР·Р°С‚РѕСЂ:adam\
    РјРµС‚СЂРёРєР°:accuracy'
    return СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_РґР»СЏ_РєР»Р°СЃСЃРёС„РёРєР°С†РёРё(СЃР»РѕРё+'-linear', РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё)
  if Р·Р°РґР°С‡Р° == 'Р°СѓРґРёРѕ':
    РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё = 'РћС€РёР±РєР°:categorical_crossentropy\
    РѕРїС‚РёРјРёР·Р°С‚РѕСЂ:adam\
    РјРµС‚СЂРёРєР°:accuracy'
    return СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_РґР»СЏ_РєР»Р°СЃСЃРёС„РёРєР°С†РёРё(СЃР»РѕРё+'-softmax', РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё)
  if Р·Р°РґР°С‡Р° == 'СЃРµРіРјРµРЅС‚Р°С†РёСЏ РёР·РѕР±СЂР°Р¶РµРЅРёР№':
    РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё = 'РћС€РёР±РєР°:categorical_crossentropy\
      РѕРїС‚РёРјРёР·Р°С‚РѕСЂ:adam\
      РјРµС‚СЂРёРєР°:dice_coef' 
    return СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_РґР»СЏ_СЃРµРіРјРµРЅС‚Р°С†РёРё(СЃР»РѕРё+'-same-softmax', РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё)
  if Р·Р°РґР°С‡Р° == 'СЃРµРіРјРµРЅС‚Р°С†РёСЏ С‚РµРєСЃС‚Р°':
    РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё = 'РћС€РёР±РєР°:categorical_crossentropy\
      РѕРїС‚РёРјРёР·Р°С‚РѕСЂ:adam\
      РјРµС‚СЂРёРєР°:dice_coef' 
    return СЃРѕР·РґР°С‚СЊ_СЃРµС‚СЊ_РґР»СЏ_СЃРµРіРјРµРЅС‚Р°С†РёРё(СЃР»РѕРё+'-same-sigmoid', РІС…РѕРґРЅРѕР№_СЂР°Р·РјРµСЂ, РїР°СЂР°РјРµС‚СЂС‹_РјРѕРґРµР»Рё)
  assert False, f'Р”Р°РЅРЅР°СЏ Р·Р°РґР°С‡Р° \'{Р·Р°РґР°С‡Р°}\' РЅРµ СЂР°СЃРїРѕР·РЅР°РЅР°. Р’РѕР·РјРѕР¶РЅРѕ РІС‹ РёРјРµР»Рё РІРІРёРґСѓ \'{check_for_errors.check_word(Р·Р°РґР°С‡Р°,"task")}\''

def РѕР±СѓС‡РµРЅРёРµ_РјРѕРґРµР»Рё_РєРІР°СЂС‚РёСЂС‹(РјРѕРґРµР»СЊ, x_train, y_train, x_test=None, y_test=None, batch_size=None, epochs=None, РєРѕСЌС„_СЂР°Р·РґРµР»РµРЅРёСЏ = 0.2):
  cur_time = time.time()
  global loss, val_loss, history, result, best_result, idx_best
  result = ''
  idx_best = 0
  best_result = 5.000
  filepath="model.h5"
  model_checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

  def on_epoch_end(epoch, log):
    global cur_time, loss, val_loss, result, best_result, idx_best, count_len
    yScaler = pickle.load(open('РёРЅСЃС‚СЂСѓРјРµРЅС‚С‹/РєРІР°СЂС‚РёСЂС‹/yscaler.pkl','rb'))
    pred = РјРѕРґРµР»СЊ.predict(x_test) #РџРѕР»СѓР°РµРј РІС‹С…РѕРґ СЃРµС‚Рё РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕ РІС‹Р±РѕСЂРєРµ
    predUnscaled = yScaler.inverse_transform(pred).flatten() #Р”РµР»Р°РµРј РѕР±СЂР°С‚РЅРѕРµ РЅРѕСЂРјРёСЂРѕРІР°РЅРёРµ РІС‹С…РѕРґР° Рє РёР·РЅР°С‡Р°Р»СЊРЅС‹Рј РІРµР»РёС‡РёРЅР°Рј С†РµРЅ РєРІР°СЂС‚РёСЂ
    yTrainUnscaled = yScaler.inverse_transform(y_test).flatten() #Р”РµР»Р°РµРј С‚Р°РєРѕРµ Р¶Рµ РѕР±СЂР°С‚РЅРѕРµ РЅРѕСЂРјРёСЂРѕРІР°РЅРёРµ yTrain Рє Р±Р°Р·РѕРІС‹Рј С†РµРЅР°Рј
    delta = predUnscaled - yTrainUnscaled #РЎС‡РёС‚Р°РµРј СЂР°Р·РЅРѕСЃС‚СЊ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ Рё РїСЂР°РІРёР»СЊРЅС‹С… С†РµРЅ
    absDelta = abs(delta) #Р‘РµСЂС‘Рј РјРѕРґСѓР»СЊ РѕС‚РєР»РѕРЅРµРЅРёСЏ

    pred2 = РјРѕРґРµР»СЊ.predict(x_train) #РџРѕР»СѓР°РµРј РІС‹С…РѕРґ СЃРµС‚Рё РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕ РІС‹Р±РѕСЂРєРµ
    predUnscaled2 = yScaler.inverse_transform(pred2).flatten() #Р”РµР»Р°РµРј РѕР±СЂР°С‚РЅРѕРµ РЅРѕСЂРјРёСЂРѕРІР°РЅРёРµ РІС‹С…РѕРґР° Рє РёР·РЅР°С‡Р°Р»СЊРЅС‹Рј РІРµР»РёС‡РёРЅР°Рј С†РµРЅ РєРІР°СЂС‚РёСЂ
    yTrainUnscaled2 = yScaler.inverse_transform(y_train).flatten() #Р”РµР»Р°РµРј С‚Р°РєРѕРµ Р¶Рµ РѕР±СЂР°С‚РЅРѕРµ РЅРѕСЂРјРёСЂРѕРІР°РЅРёРµ yTrain Рє Р±Р°Р·РѕРІС‹Рј С†РµРЅР°Рј
    delta2 = predUnscaled2 - yTrainUnscaled2 #РЎС‡РёС‚Р°РµРј СЂР°Р·РЅРѕСЃС‚СЊ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ Рё РїСЂР°РІРёР»СЊРЅС‹С… С†РµРЅ
    absDelta2 = abs(delta2) #Р‘РµСЂС‘Рј РјРѕРґСѓР»СЊ РѕС‚РєР»РѕРЅРµРЅРёСЏ
    loss.append(sum(absDelta2) / (1e+6 * len(absDelta2)))
    val_loss.append(sum(absDelta) / (1e+6 * len(absDelta)))
    
    p1 = 'Р­РїРѕС…Р° в„–' + str(epoch+1)
    p2 = p1 + ' '* (10 - len(p1)) + 'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: ' + str(round(time.time()-cur_time,2)) +'c'
    p3 = p2 + ' '* (33 - len(p2)) + 'РћС€РёР±РєР° РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(sum(absDelta2) / (1e+6 * len(absDelta2)), 3))+'РјР»РЅ'
    p4 = p3 + ' '* (77 - len(p3)) + 'РћС€РёР±РєР° РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(sum(absDelta) / (1e+6 * len(absDelta)), 3))+'РјР»РЅ'
    result += p4 + '\n' 
    
    a = round(sum(absDelta) / (1e+6 * len(absDelta)), 3)
    if a < best_result:
        best_result = a
        idx_best = epoch
    print(p4)
  
    # РљРѕР»Р»Р±СЌРєРё

  def on_train_begin(log):
    global cur_time, loss, val_loss
    loss=[]
    val_loss = []

  def on_epoch_begin(epoch, log):
    global cur_time
    cur_time = time.time()

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-6)

  myCB = LambdaCallback(on_train_begin=on_train_begin, on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
  myCB23 = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
  РјРѕРґРµР»СЊ.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB, reduce_lr], verbose = 0)
  РјРѕРґРµР»СЊ.load_weights('model.h5')
  РјРѕРґРµР»СЊ.save('model_s.h5')
#   РјРѕРґРµР»СЊ.compile(optimizer=Nadam(lr=1e-4), loss='mae')
#   РјРѕРґРµР»СЊ.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB23], verbose = 0)
#   РјРѕРґРµР»СЊ.load_weights('model.h5')
#   РјРѕРґРµР»СЊ.compile(optimizer=Nadam(lr=1e-5), loss='mae')
#   РјРѕРґРµР»СЊ.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB23], verbose = 0)
#   РјРѕРґРµР»СЊ.load_weights('model.h5')
#   РјРѕРґРµР»СЊ.save('model_s.h5')

  clear_output(wait=True)
  result = result.split('\n')
  
  idx = 0
  for i in range(len(result)):
    if str(best_result) in result[i]:
      idx = i

  for i in range(len(result)):
    s = result[i]
    if i == idx:
      s = colored(result[i], color='white', on_color='on_green')
    print(s)
      
  plt.figure(figsize=(12, 6)) # РЎРѕР·РґР°РµРј РїРѕР»РѕС‚РЅРѕ РґР»СЏ РІРёР·СѓР°Р»РёР·Р°С†РёРё  
  plt.plot(loss, label ='РћР±СѓС‡Р°СЋС‰Р°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє РѕС€РёР±РєРё РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ
  plt.plot(val_loss, label ='РџСЂРѕРІРµСЂРѕС‡РЅР°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє РѕС€РёР±РєРё РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ
  plt.legend() # Р’С‹РІРѕРґРёРј РїРѕРґРїРёСЃРё РЅР° РіСЂР°С„РёРєРµ
  plt.title('Р“СЂР°С„РёРє РѕС€РёР±РєРё РѕР±СѓС‡РµРЅРёСЏ') # Р’С‹РІРѕРґРёРј РЅР°Р·РІР°РЅРёРµ РіСЂР°С„РёРєР°
  plt.show()
  
def РѕР±СѓС‡РµРЅРёРµ_РјРѕРґРµР»Рё_С‚СЂР°С„РёРє(РјРѕРґ, РіРµРЅ1, РіРµРЅ2, РєРѕР»РёС‡РµСЃС‚РІРѕ_СЌРїРѕС…=None):
  global result, idx_best, best_result, history
  result = ''
  idx_best = 0
  best_result = 1000000
  filepath="model.h5"
  model_checkpoint_callback = ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    
  cur_time = time.time()

  def on_epoch_end(epoch, log):
    k = list(log.keys())
    global cur_time, result, idx_best, best_result
    p1 = 'Р­РїРѕС…Р° в„–' + str(epoch+1)
    p2 = p1 + ' '* (10 - len(p1)) + 'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: ' + str(round(time.time()-cur_time,2)) +'c'
    p3 = p2 + ' '* (33 - len(p2)) + 'РћС€РёР±РєР° РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(log[k[0]],5))
    p4 = p3 + ' '* (77 - len(p3)) + 'РћС€РёР±РєР° РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(log[k[2]],5))
    result += p4 + '\n'
    if log[k[2]] < best_result:
        best_result = log[k[2]]
        idx_best = epoch
    print(p4)
    cur_time = time.time()

  def on_epoch_begin(epoch, log):
    global cur_time
    cur_time = time.time()

  myCB = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)

  history = РјРѕРґ.fit_generator(РіРµРЅ1, epochs=РєРѕР»РёС‡РµСЃС‚РІРѕ_СЌРїРѕС…, verbose=0, validation_data=РіРµРЅ2, callbacks=[model_checkpoint_callback, myCB])
  clear_output(wait=True)
  result = result.split('\n')
  for i in range(len(result)):
    s = result[i]
    if i == idx_best:
      s = colored(result[i], color='white', on_color='on_green')
    print(s)
  plt.plot(history.history['loss'], label='РћС€РёР±РєР° РЅР° РѕР±СѓС‡Р°СЋС‰РµРј РЅР°Р±РѕСЂРµ')
  plt.plot(history.history['val_loss'], label='РћС€РёР±РєР° РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕРј РЅР°Р±РѕСЂРµ')
  plt.ylabel('РЎСЂРµРґРЅСЏСЏ РѕС€РёР±РєР°')
  plt.legend()

def РѕР±СѓС‡РµРЅРёРµ_РјРѕРґРµР»Рё(РјРѕРґРµР»СЊ, x_train, y_train, x_test=[], y_test=[], 
                batch_size=None, epochs=None, РєРѕСЌС„_СЂР°Р·РґРµР»РµРЅРёСЏ = 0.2, **kwargs):
  global result, idx_best, stage_history, best_result, history, epoch, stage_result, stage_best_result, best_result_on_train_arr
  cur_time = time.time()
  if 'РєРѕР»РёС‡РµСЃС‚РІРѕ_Р·Р°РїСѓСЃРєРѕРІ' in kwargs:
    cur_time = time.time()
    global stage_result
    stage_result = ''
    stage_best_result = []
    best_result_on_train_arr = []
    stage_history = []
    for start in range(kwargs['РєРѕР»РёС‡РµСЃС‚РІРѕ_Р·Р°РїСѓСЃРєРѕРІ']):
      result = ''
      idx_best = 0
      best_result = 0
      best_result_on_train = 0
      if batch_size == None:
        batch_size = 16
      if epochs == None:
        epochs = 10
      filepath="model.h5"

      model_checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
        
      cur_time = time.time()
      def on_epoch_end(epoch, log):
        global cur_time, result, best_result, idx_best, stage_result, С‚РµРєСѓС‰Р°СЏ_СЌРїРѕС…Р°, stage_best_result, best_result_on_train, best_result_on_train_arr
        С‚РµРєСѓС‰Р°СЏ_СЌРїРѕС…Р° = epoch
        k = list(log.keys()) 
        # p1 = 'Р—Р°РїСѓСЃРє в„–' + str(start+1)
        # p3 = p1 + ' ' * (10 - len(p1)) + ' РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(log[k[1]]*100,2))+'%'
        
        p1 = 'Р—Р°РїСѓСЃРє в„–' + str(start+1)
        p2 = p1 + ' '* (12 - len(p1)) + 'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: ' + str(round((time.time()-cur_time) * epochs,2)) +'c'
        p3 = p2 + ' '* (34 - len(p2)) + 'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(log[k[1]]*100,2))+'%'
        if len(k)>2:
            p4 = p3 + ' '* (77 - len(p3)) + 'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(log[k[3]]*100,2))+'%'
            result += p4 + '\n'
            if log[k[3]]*100 >= best_result:
              best_result = log[k[3]]*100
              best_result_on_train = log[k[1]]*100
              idx_best = epoch
        else:
            result += p3 + '\n'
            if log[k[1]]*100 >= best_result:
              best_result = log[k[1]]*100
              best_result_on_train = log[k[1]]*100
              idx_best = epoch 

        if (С‚РµРєСѓС‰Р°СЏ_СЌРїРѕС…Р°+1) == epochs:
          result = result.split('\n')
          stage_best_result.append(best_result)
          best_result_on_train_arr.append(best_result_on_train)
          for i in range(len(result)):
            if i == idx_best:
              best_in_epoc = result[i]
              stage_result += best_in_epoc + '\n'
          print(best_in_epoc)
        cur_time = time.time()

      def on_epoch_begin(epoch, log):
        global cur_time
        cur_time = time.time()

      myCB = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
      
      if len(x_test)==0:
        model_checkpoint_callback = ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)
        history = РјРѕРґРµР»СЊ.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint_callback, myCB], verbose = 0)
        stage_history.append(history)
      else:
        history = РјРѕРґРµР»СЊ.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB], verbose = 0)
        stage_history.append(history)
      РјРѕРґРµР»СЊ.load_weights('model.h5')
      РјРѕРґРµР»СЊ.save('model_s.h5')

    clear_output(wait=True)
    stage_result = stage_result.split('\n')
    stage_best_result = np.asarray(stage_best_result)
    best_result_on_train_arr = np.asarray(best_result_on_train_arr)
    idx_best_stage = np.argmax(stage_best_result)

    for i in range(len(stage_result)):
      s = stage_result[i]
      if i == idx_best_stage:
        s = colored(stage_result[i], color='white', on_color='on_green')
      print(s)
    print('РЎСЂРµРґРЅСЏСЏ С‚РѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ:   ' , str(round(np.mean(best_result_on_train_arr), 2)) +  '%', '\n' 
          'РЎСЂРµРґРЅСЏСЏ С‚РѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: ' , str(round(np.mean(stage_best_result), 2)) + '%')

    
    plt.figure(figsize=(12,6)) # РЎРѕР·РґР°РµРј РїРѕР»РѕС‚РЅРѕ РґР»СЏ РІРёР·СѓР°Р»РёР·Р°С†РёРё
    keys = list(stage_history[idx_best_stage].history.keys())
    plt.plot(stage_history[idx_best_stage].history[keys[1]], label ='РћР±СѓС‡Р°СЋС‰Р°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє С‚РѕС‡РЅРѕСЃС‚Рё РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ
    if len(keys)>2:
      plt.plot(stage_history[idx_best_stage].history['val_'+keys[1]], label ='РџСЂРѕРІРµСЂРѕС‡РЅР°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє С‚РѕС‡РЅРѕСЃС‚Рё РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ
    plt.legend() # Р’С‹РІРѕРґРёРј РїРѕРґРїРёСЃРё РЅР° РіСЂР°С„РёРєРµ
    plt.title('Р“СЂР°С„РёРє С‚РѕС‡РЅРѕСЃС‚Рё РѕР±СѓС‡РµРЅРёСЏ, РЅР° Р»СѓС‡С€РµРј Р·Р°РїСѓСЃРєРµ') # Р’С‹РІРѕРґРёРј РЅР°Р·РІР°РЅРёРµ РіСЂР°С„РёРєР°
    plt.show()
    return history


  else:
    result = ''
    idx_best = 0
    best_result = 0
    if batch_size == None:
      batch_size = 16
    if epochs == None:
      epochs = 10
    filepath="model.h5"
    model_checkpoint_callback = ModelCheckpoint(
      filepath=filepath,
      save_weights_only=True,
      monitor='val_loss',
      mode='min',
      save_best_only=True)
      
    cur_time = time.time()
    def on_epoch_end(epoch, log):
      k = list(log.keys())
      global cur_time, result, idx_best, best_result   
      p1 = 'Р­РїРѕС…Р° в„–' + str(epoch+1)
      p2 = p1 + ' '* (12 - len(p1)) + 'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: ' + str(round(time.time()-cur_time,2)) +'c'
      p3 = p2 + ' '* (36 - len(p2)) + 'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(log[k[1]]*100,2))+'%'
      if len(k)>2:
          p4 = p3 + ' '* (77 - len(p3)) + 'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: ' + str(round(log[k[3]]*100,2))+'%'
          result += p4 + '\n'
          if log[k[3]]*100 >= best_result:
            best_result = log[k[3]]*100
            idx_best = epoch
          print(p4)
      else:
          result += p3 + '\n'
          if log[k[1]]*100 >= best_result:
            best_result = log[k[1]]*100
            idx_best = epoch
          print(p3)    
      cur_time = time.time()   

    def on_epoch_begin(epoch, log):
      global cur_time
      cur_time = time.time()

    myCB = LambdaCallback(on_epoch_end = on_epoch_end, on_epoch_begin=on_epoch_begin)
    
    if len(x_test)==0:
      model_checkpoint_callback = ModelCheckpoint(
          filepath=filepath,
          save_weights_only=True,
          monitor='loss',
          mode='min',
          save_best_only=True)
      history = РјРѕРґРµР»СЊ.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[model_checkpoint_callback, myCB], verbose = 0)
    else:
      history = РјРѕРґРµР»СЊ.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[model_checkpoint_callback, myCB], verbose = 0)
    
    РјРѕРґРµР»СЊ.load_weights('model.h5')
    РјРѕРґРµР»СЊ.save('model_s.h5')
    clear_output(wait=True)
    result = result.split('\n')
    for i in range(len(result)):
      s = result[i]
      if i == idx_best:
        s = colored(result[i], color='white', on_color='on_green')
      print(s)
    plt.figure(figsize=(12,6)) # РЎРѕР·РґР°РµРј РїРѕР»РѕС‚РЅРѕ РґР»СЏ РІРёР·СѓР°Р»РёР·Р°С†РёРё
    keys = list(history.history.keys())
    plt.plot(history.history[keys[1]], label ='РћР±СѓС‡Р°СЋС‰Р°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє С‚РѕС‡РЅРѕСЃС‚Рё РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ
    if len(keys)>2:
      plt.plot(history.history['val_'+keys[1]], label ='РџСЂРѕРІРµСЂРѕС‡РЅР°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє С‚РѕС‡РЅРѕСЃС‚Рё РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ
    plt.legend() # Р’С‹РІРѕРґРёРј РїРѕРґРїРёСЃРё РЅР° РіСЂР°С„РёРєРµ
    plt.title('Р“СЂР°С„РёРє С‚РѕС‡РЅРѕСЃС‚Рё РѕР±СѓС‡РµРЅРёСЏ') # Р’С‹РІРѕРґРёРј РЅР°Р·РІР°РЅРёРµ РіСЂР°С„РёРєР°
    plt.show()
    return history

def СЃРѕР·РґР°С‚СЊ_РІС‹Р±РѕСЂРєРё_РІР°РєР°РЅСЃРёРё():
  global idx_val
  dir = 'РІР°РєР°РЅСЃРёРё'
  """
  Parameters:
  dir (str)              РџСѓС‚СЊ Рє РїР°РїРєРµ СЃ РґР°С‚Р°СЃРµС‚РѕРј

  Return:
  x_train, y_train, x_val, y_val
  """

  data = np.load(dir + '/X_train.npy')
  labels = np.load(dir + '/Y_train.npy')

  x_train, x_val, y_train, y_val = data[:500], data[300:], labels[:500], labels[300:]
  idx_val = [i for i in range(data[300:500].shape[0])]
  
  if not os.path.exists('test_sets'):
      os.mkdir('test_sets')
  np.save(f'test_sets/РІР°РєР°РЅСЃРёРё.npy', x_val)
  np.save(f'test_sets/РІР°РєР°РЅСЃРёРё_РјРµС‚РєРё.npy', y_val)

  return (x_train, y_train), (x_val, y_val)

def С‚РµСЃС‚_РјРѕРґРµР»Рё_РІР°РєР°РЅСЃРёРё(РЅРµР№СЂРѕРЅРєР°, x_val, y_val):
    dir='РІР°РєР°РЅСЃРёРё'
    num = 3
    data = pd.read_csv(dir + '/data.csv')
    data = data.iloc[300:500]
    data.reset_index(inplace=True, drop=True)

    columns=['РџРѕР»', 'Р’РѕР·СЂР°СЃС‚', 'Р“РѕСЂРѕРґ', 'Р“РѕС‚РѕРІРЅРѕСЃС‚СЊ Рє РїРµСЂРµРµР·РґСѓ',
             'Р“РѕС‚РѕРІРЅРѕСЃС‚СЊ Рє РєРѕРјР°РЅРґРёСЂРѕРІРєР°Рј', 'Р“СЂР°Р¶РґР°РЅСЃС‚РІРѕ', 'Р Р°Р·СЂРµС€РµРЅРёРµ РЅР° СЂР°Р±РѕС‚Сѓ',
             'Р—РЅР°РЅРёСЏ СЏР·С‹РєРѕРІ', 'РћР±СЂР°Р·РѕРІР°РЅРёРµ', 'Р”РѕРїРѕР»РЅРёС‚РµР»СЊРЅРѕРµ РѕР±СЂР°Р·РѕРІР°РЅРёРµ',
             'Р—Р°СЂРїР»Р°С‚Р°', 'Р’СЂРµРјСЏ РІ РїСѓС‚Рё РґРѕ СЂР°Р±РѕС‚С‹', 'Р—Р°РЅСЏС‚РѕСЃС‚СЊ', 
             'Р“СЂР°С„РёРє', 'РћРїС‹С‚ СЂР°Р±РѕС‚С‹ (РјРµСЃ)', 'РћР±СЏР·Р°РЅРЅРѕСЃС‚Рё РЅР° РїСЂРµРґ.СЂР°Р±РѕС‚Рµ']

    idx = np.random.randint(0, len(idx_val), num)

    for i in idx:
        pred = РЅРµР№СЂРѕРЅРєР°.predict(np.expand_dims(x_val[i], axis=0))

        print('РўРµСЃС‚РѕРІРѕРµ СЂРµР·СЋРјРµ:')
        print()

        for column in columns:
            info = str(data[column][i])
            info = info[:600]
            if info == 'no_data':
                info = 'Р”Р°РЅРЅС‹Рµ РЅРµ СѓРєР°Р·Р°РЅС‹'
            step = 100
            for j in range(0, len(info), step):
                if j == 0:
                    print('%-28s' % (column + ':'), info[j:j+step])
                else:
                    print('%-28s' % (''), info[j:j+step])
        print()
        print(f'РњРѕРґРµР»СЊ СѓРІРµСЂРµРЅР°, С‡С‚Рѕ РєР°РЅРґРёРґР°С‚ РїРѕРґС…РѕРґРёС‚ РІ РЈРР РЅР°: {round(100 - 100 * pred[0,1], 2)}%')
        print('---------------------------------------------------------------')
        result = data['Р­С‚Р°Рї СЃРґРµР»РєРё'][i]
        print('%-28s' % 'РџРѕРґС…РѕРґРёС‚ Р»Рё РєР°РЅРґРёРґР°С‚ РЅР° СЃР°РјРѕРј РґРµР»Рµ: ', result)
        print('---------------------------------------------------------------')
        print()


def С‚РµСЃС‚_РјРѕРґРµР»Рё_РєР»Р°СЃСЃРёС„РёРєР°С†РёРё(РјРѕРґРµР»СЊ=None, С‚РµСЃС‚РѕРІС‹Р№_РЅР°Р±РѕСЂ=None, РїСЂР°РІРёР»СЊРЅС‹Рµ_РѕС‚РІРµС‚С‹=[], РєР»Р°СЃСЃС‹=None, РєРѕР»РёС‡РµСЃС‚РІРѕ=1):
  for i in range(РєРѕР»РёС‡РµСЃС‚РІРѕ):
    number = np.random.randint(С‚РµСЃС‚РѕРІС‹Р№_РЅР°Р±РѕСЂ.shape[0]) # Р—Р°РґР°РµРј РёРЅРґРµРєСЃ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ РІ С‚РµСЃС‚РѕРІРѕРј РЅР°Р±РѕСЂРµ
    sample = С‚РµСЃС‚РѕРІС‹Р№_РЅР°Р±РѕСЂ[number]
    if sample.shape == (784,):
      sample = sample.reshape((28,28))  
    if sample.shape == (28, 28, 1):
      sample = sample.reshape((28,28))
    print('РўРµСЃС‚РѕРІРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёРµ:')
    plt.imshow(sample, cmap='gray') # Р’С‹РІРѕРґРёРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ РёР· С‚РµСЃС‚РѕРІРѕРіРѕ РЅР°Р±РѕСЂР° СЃ Р·Р°РґР°РЅРЅС‹Рј РёРЅРґРµРєСЃРѕРј
    plt.axis('off') # РћС‚РєР»СЋС‡Р°РµРј РѕСЃРё
    plt.show() 

    sample = С‚РµСЃС‚РѕРІС‹Р№_РЅР°Р±РѕСЂ[number].reshape((1 + РјРѕРґРµР»СЊ.input.shape[1:]))
    pred = РјРѕРґРµР»СЊ.predict(sample)[0] # Р Р°СЃРїРѕР·РЅР°РµРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ СЃ РїРѕРјРѕС‰СЊСЋ РѕР±СѓС‡РµРЅРЅРѕР№ РјРѕРґРµР»Рё
    print()
    print('Р РµР·СѓР»СЊС‚Р°С‚ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ РјРѕРґРµР»Рё:')

    def out_red(text):
      return "\033[4m\033[31m\033[31m{}\033[0m".format(text)

    def out_green(text):
      return "\033[4m\033[32m\033[32m{}\033[0m".format(text)

    def keywithmaxval(d):
      global r
      v = list(d.values())
      k = list(d.keys())
      return str(k[v.index(max(v))])
      

    dicts = {}

    for i in range(len(РєР»Р°СЃСЃС‹)):
      dicts[РєР»Р°СЃСЃС‹[i]] = round(100*pred[i],2)
      print('РњРѕРґРµР»СЊ СЂР°СЃРїРѕР·РЅР°Р»Р° РјРѕРґРµР»СЊ ',РєР»Р°СЃСЃС‹[i],' РЅР° ',round(100*pred[i],2),'%',sep='')
    print('---------------------------')

    answer = str(РєР»Р°СЃСЃС‹[РїСЂР°РІРёР»СЊРЅС‹Рµ_РѕС‚РІРµС‚С‹[number]])

    if len(РїСЂР°РІРёР»СЊРЅС‹Рµ_РѕС‚РІРµС‚С‹)>0:
      if keywithmaxval(dicts) == answer:
        print('РџСЂР°РІРёР»СЊРЅС‹Р№ РѕС‚РІРµС‚: ', out_green(answer))
        print('---------------------------')
        print()
        print()
      
      elif keywithmaxval(dicts) != РєР»Р°СЃСЃС‹[РїСЂР°РІРёР»СЊРЅС‹Рµ_РѕС‚РІРµС‚С‹[number]]:
        print('РџСЂР°РІРёР»СЊРЅС‹Р№ РѕС‚РІРµС‚: ', out_red(answer))
        print('---------------------------')
        print()
        print()

def С‚РµСЃС‚_РјРѕРґРµР»Рё_С‚СЂРµРєРµСЂ(РјРѕРґРµР»СЊ=None, С‚РµСЃС‚РѕРІС‹Р№_РЅР°Р±РѕСЂ=None, РїСЂР°РІРёР»СЊРЅС‹Рµ_РѕС‚РІРµС‚С‹=[], РєР»Р°СЃСЃС‹= ['not_same', 'same'], РєРѕР»РёС‡РµСЃС‚РІРѕ=1):
  classes = РєР»Р°СЃСЃС‹
  РєРѕР»РёС‡РµСЃС‚РІРѕ = РєРѕР»РёС‡РµСЃС‚РІРѕ
  С‚РµСЃС‚РѕРІР°СЏ_РІС‹Р±РѕСЂРєР° = С‚РµСЃС‚РѕРІС‹Р№_РЅР°Р±РѕСЂ
  predict = РјРѕРґРµР»СЊ.predict(С‚РµСЃС‚РѕРІР°СЏ_РІС‹Р±РѕСЂРєР°)
  fig = plt.figure(figsize=(16,7))
  for i, idx in enumerate(np.random.randint(0, predict.shape[0]-1, РєРѕР»РёС‡РµСЃС‚РІРѕ)):
      concat_img = С‚РµСЃС‚РѕРІР°СЏ_РІС‹Р±РѕСЂРєР°[idx]
      label = np.argmax(predict[idx])
      first_img = concat_img[:,:,:3]
      second_img = concat_img[:,:,3:]
      ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
      ax.set_title(classes[label])
      plt.imshow(first_img)

      ax = fig.add_subplot(2, 5, i+6, xticks=[], yticks=[])
      plt.imshow(second_img)

def С‚РµСЃС‚_РЅР°_СЃРІРѕРµРј_РёР·РѕР±СЂР°Р¶РµРЅРёРё(РЅРµР№СЂРѕРЅРєР°, СЂР°Р·РјРµСЂ_РёР·РѕР±СЂР°Р¶РµРЅРёСЏ, РєР»Р°СЃСЃС‹):
  fname = files.upload()
  fname = list(fname.keys())[0]
  sample = image.load_img(''+ fname, target_size=(СЂР°Р·РјРµСЂ_РёР·РѕР±СЂР°Р¶РµРЅРёСЏ[0], СЂР°Р·РјРµСЂ_РёР·РѕР±СЂР°Р¶РµРЅРёСЏ[1])) # Р—Р°РіСЂСѓР¶Р°РµРј РєР°СЂС‚РёРЅРєСѓ
  img_numpy = np.array(sample)[None,...] # РџСЂРµРѕР±СЂР°Р·СѓРµРј Р·РѕР±СЂР°Р¶РµРЅРёРµ РІ numpy-РјР°СЃСЃРёРІ
  img_numpy = img_numpy/255

  number = np.random.randint(img_numpy.shape[0]) # Р—Р°РґР°РµРј РёРЅРґРµРєСЃ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ РІ С‚РµСЃС‚РѕРІРѕРј РЅР°Р±РѕСЂРµ
  sample = img_numpy[number]
  if sample.shape == (784,):
    sample = sample.reshape((28,28))  
  if sample.shape == (28, 28, 1):
    sample = sample.reshape((28,28))
  print('РўРµСЃС‚РѕРІРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёРµ:')
  plt.imshow(sample, cmap='gray') # Р’С‹РІРѕРґРёРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ РёР· С‚РµСЃС‚РѕРІРѕРіРѕ РЅР°Р±РѕСЂР° СЃ Р·Р°РґР°РЅРЅС‹Рј РёРЅРґРµРєСЃРѕРј
  plt.axis('off') # РћС‚РєР»СЋС‡Р°РµРј РѕСЃРё
  plt.show() 

  sample = img_numpy[number].reshape((1 + РЅРµР№СЂРѕРЅРєР°.input.shape[1:]))
  pred = РЅРµР№СЂРѕРЅРєР°.predict(sample)[0] # Р Р°СЃРїРѕР·РЅР°РµРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ СЃ РїРѕРјРѕС‰СЊСЋ РѕР±СѓС‡РµРЅРЅРѕР№ РјРѕРґРµР»Рё
  print()
  print('Р РµР·СѓР»СЊС‚Р°С‚ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ РјРѕРґРµР»Рё:')

  def keywithmaxval(d):
    global r
    v = list(d.values())
    k = list(d.keys())
    return str(k[v.index(max(v))])
    
  dicts = {}

  for i in range(len(РєР»Р°СЃСЃС‹)):
    dicts[РєР»Р°СЃСЃС‹[i]] = round(100*pred[i],2)
    print('РњРѕРґРµР»СЊ СЂР°СЃРїРѕР·РЅР°Р»Р° РјРѕРґРµР»СЊ ',РєР»Р°СЃСЃС‹[i],' РЅР° ',round(100*pred[i],2),'%',sep='')

  print('РќРµР№СЂРѕРЅРЅР°СЏ СЃРµС‚СЊ СЃС‡РёС‚Р°РµС‚, С‡С‚Рѕ СЌС‚Рѕ: ', keywithmaxval(dicts))
    
def С‚РµСЃС‚_РјРѕРґРµР»Рё_HR(gan_generator):
  def load_image(path):
    return np.array(Image.open(path))
  #Р¤СѓРЅРєС†РёСЏ РґР»СЏ РїСЂРµРѕР±СЂР°Р·РѕРІР°РЅРёСЏ РєР°СЂС‚РёРЅРєРё lr РІ sr
  def resolve(model, lr_batch):
      lr_batch = tf.cast(lr_batch, tf.float32)
      sr_batch = model(lr_batch)
      sr_batch = tf.clip_by_value(sr_batch, 0, 255)
      sr_batch = tf.round(sr_batch)
      sr_batch = tf.cast(sr_batch, tf.uint8)
      return sr_batch 
  def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]
  def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)    
    gan_sr = resolve_single(gan_generator, lr)    
    plt.figure(figsize=(10, 15))
    images = [lr, gan_sr]
    titles = ['РСЃС…РѕРґРЅРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёРµ', 'РР·РѕР±СЂР°Р¶РµРЅРёРµ РїРѕСЃР»Рµ РѕР±СЂР°Р±РѕС‚РєРё']
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.xticks([])
        plt.yticks([])  

  url = 'https://storage.googleapis.com/aiu_bucket/Examples.zip' # РЈРєР°Р·С‹РІР°РµРј URL-С„Р°Р№Р»Р°
  output = 'Examples.zip' # РЈРєР°Р·С‹РІР°РµРј РёРјСЏ С„Р°Р№Р»Р°, РІ РєРѕС‚РѕСЂС‹Р№ СЃРѕС…СЂР°РЅСЏРµРј С„Р°Р№Р»
  gdown.download(url, output, quiet=True) # РЎРєР°С‡РёРІР°РµРј С„Р°Р№Р» РїРѕ СѓРєР°Р·Р°РЅРЅРѕРјСѓ URL
  # РЎРєР°С‡РёРІР°РµРј Рё СЂР°СЃРїР°РєРѕРІС‹РІР°РµРј Р°СЂС…РёРІ
  РґР°С‚Р°СЃРµС‚.СЂР°СЃРїР°РєРѕРІР°С‚СЊ_Р°СЂС…РёРІ(
      РѕС‚РєСѓРґР° = "Examples.zip",
      РєСѓРґР° = "/content"
  )
  for file in os.listdir('demo1/'):
    resolve_and_plot('demo1/' + file)

def РїРѕРєР°Р·Р°С‚СЊ_РіСЂР°С„РёРє_РѕР±СѓС‡РµРЅРёСЏ(**kwargs):
  keys = list(kwargs['СЃС‚Р°С‚РёСЃС‚РёРєР°'].history.keys())
  for i in range(len(keys)//2):
    plt.figure(figsize=(12, 6)) # РЎРѕР·РґР°РµРј РїРѕР»РѕС‚РЅРѕ РґР»СЏ РІРёР·СѓР°Р»РёР·Р°С†РёРё
    plt.plot(kwargs['СЃС‚Р°С‚РёСЃС‚РёРєР°'].history[keys[i]], label ='РћР±СѓС‡Р°СЋС‰Р°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє РѕС€РёР±РєРё РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ
    plt.plot(kwargs['СЃС‚Р°С‚РёСЃС‚РёРєР°'].history['val_'+keys[i]], label ='РџСЂРѕРІРµСЂРѕС‡РЅР°СЏ РІС‹Р±РѕСЂРєР°') # Р’РёР·СѓР°Р»РёР·РёСЂСѓРµРј РіСЂР°С„РёРє РѕС€РёР±РєРё РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ
    plt.legend() # Р’С‹РІРѕРґРёРј РїРѕРґРїРёСЃРё РЅР° РіСЂР°С„РёРєРµ
    if 'loss' in keys[i]:
      plt.title('Р“СЂР°С„РёРє РѕС€РёР±РєРё РѕР±СѓС‡РµРЅРёСЏ РјРѕРґРµР»Рё') # Р’С‹РІРѕРґРёРј РЅР°Р·РІР°РЅРёРµ РіСЂР°С„РёРєР°
    else:
      plt.title('Р“СЂР°С„РёРє С‚РѕС‡РЅРѕСЃС‚Рё РѕР±СѓС‡РµРЅРёСЏ РјРѕРґРµР»Рё') # Р’С‹РІРѕРґРёРј РЅР°Р·РІР°РЅРёРµ РіСЂР°С„РёРєР°
    plt.show()

def Р·Р°РіСЂСѓР·РёС‚СЊ_РїСЂРµРґРѕР±СѓС‡РµРЅРЅСѓСЋ_РјРѕРґРµР»СЊ():
  url = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1QYLIUQWWyqLvn8TCEAZiY7q8umv7lyYw' # РЈРєР°Р·С‹РІР°РµРј URL-С„Р°Р№Р»Р°
  output = 'model.h5' # РЈРєР°Р·С‹РІР°РµРј РёРјСЏ С„Р°Р№Р»Р°, РІ РєРѕС‚РѕСЂС‹Р№ СЃРѕС…СЂР°РЅСЏРµРј С„Р°Р№Р»
  gdown.download(url, output, quiet=True) 
  model = load_model('model.h5')
  return model

def СЃРѕР·РґР°С‚СЊ_РјРѕРґРµР»СЊ_HighResolution():
  LR_SIZE = 24
  HR_SIZE = 96
    #РљРѕСЌС„С„РёС†РёРµРЅС‚С‹ РґР»СЏ РїСЂРµРѕР±СЂР°Р·РѕРІР°РЅРёСЏ RGB
  DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
  # РќРѕСЂРјР°Р»РёР·СѓРµС‚ RGB РёР·РѕР±СЂР°Р¶РµРЅРёСЏ Рє РїСЂРѕРјРµР¶СѓС‚РєСѓ [0, 1]
  def normalize_01(x):
    return x / 255.0
  # res_block
  def res_block(x_in, num_filters, momentum=0.8):
      x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
      x = BatchNormalization(momentum=momentum)(x)
      x = PReLU(shared_axes=[1, 2])(x)
      x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
      x = BatchNormalization(momentum=momentum)(x)
      x = Add()([x_in, x])
      return x  
  # Р‘Р»РѕРє Р°РїСЃРµРјРїР»РёРЅРіР°
  def upsample(x_in, num_filters):
      x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
      x = Lambda(pixel_shuffle(scale=2))(x)
      return PReLU(shared_axes=[1, 2])(x)   

  def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)     

  # РћР±СЂР°С‚РЅР°СЏ РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ
  def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean
  # РћР±СЂР°С‚РЅР°СЏ РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ
  def denormalize_m11(x):
      return (x + 1) * 127.5
      
  def sr_resnet(num_filters=64, num_res_blocks=16):
      x_in = Input(shape=(None, None, 3))
      x = Lambda(normalize_01)(x_in)

      x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
      x = x_1 = PReLU(shared_axes=[1, 2])(x)

      for _ in range(num_res_blocks):
          x = res_block(x, num_filters)

      x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
      x = BatchNormalization()(x)
      x = Add()([x_1, x])

      x = upsample(x, num_filters * 4)
      x = upsample(x, num_filters * 4)

      x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
      x = Lambda(denormalize_m11)(x)

      return Model(x_in, x)
  generator = sr_resnet
  return generator

#LabStory
#РђРІС‚РѕСЂРёР·Р°С†РёСЏ
def Р°РІС‚РѕСЂРёР·Р°С†РёСЏ_LabStory():
  global token
  global headers
  global user_id

  login = input('Р’РІРµРґРёС‚Рµ Р»РѕРіРёРЅ:')
  password = getpass.getpass('Р’РІРµРґРёС‚Рµ РїР°СЂРѕР»СЊ:')
  headers = {"content-type": "application/json"} # Р¤РѕСЂРјРёСЂСѓРµС‚ Р·Р°РіРѕР»РѕРІРѕРє (СѓРєР°Р·С‹РІР°РµРј, С‡С‚Рѕ С‚РёРї РєРѕРЅС‚РµРЅС‚Р° json)
  post = 'http://labstory.neural-university.ru/api/login?email='+login+'&password='+password # Р¤РѕСЂРјРёСЂСѓРµС‚ post-Р·Р°РїСЂРѕСЃ
  json_response = requests.post(post, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ

  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    print('РђРІС‚РѕСЂРёР·Р°С†РёСЏ СѓСЃРїРµС€РЅРѕ Р·Р°РІРµСЂС€РµРЅР°')
    token = ast.literal_eval(json_response.text)['token'] # РЎРѕС…СЂР°РЅСЏРµРј С‚РѕРєРµРЅ (ast.literal_eval - text to dict)
  elif json_response.status_code == 422: # Р•СЃР»Рё РїСЂРёС€РµР» 422-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    print('РќРµРІРµСЂРЅС‹Р№ Р»РѕРіРё РёР»Рё РїР°СЂРѕР»СЊ.')
    return
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
  get = 'http://labstory.neural-university.ru/api/my/profile?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    user = json_response.json()
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
  user_id = user['id']

# Р”РѕР±Р°РІР»РµРЅРёРµ РЅР°Р±РѕСЂР° РґР°РЅРЅС‹С…
def РґРѕР±Р°РІРёС‚СЊ_РґР°С‚Р°СЃРµС‚_LabStory(dataset_dict):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  global РґР°С‚Р°СЃРµС‚_id
  post = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token

  name = dataset_dict['name']
  description = dataset_dict['description']
  url = dataset_dict['url']
  data = {
      "url": url,
      "name": name,
      "description": description
  }
  json_response = requests.post(post, json=data, headers=headers)
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    # РІС‹С‡РёСЃР»РµРЅРёРµ id РґР°С‚Р°СЃРµС‚Р° РїСѓС‚РµРј РїРѕР»СѓС‡РµРЅРёСЏ СЃРїРёСЃРєР° РІСЃРµС… РґР°С‚Р°СЃРµС‚РѕРІ Рё РїРѕР»СѓС‡РµРЅРёСЏ id РїРѕСЃР»РµРґРЅРµРіРѕ
    get = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token
    json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
    datasets = json_response.json()
    meta = datasets['meta']
    get = 'http://labstory.neural-university.ru/api/my/datasets?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    datasets = json_response.json()
    d = datasets['data'][-1]
    РґР°С‚Р°СЃРµС‚_id = d["id"]
    print('Р”Р°С‚Р°СЃРµС‚ СѓСЃРїРµС€РЅРѕ РґРѕР±Р°РІР»РµРЅ')
    print(f'id: {d["id"]}')
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
  
# РџРѕР»СѓС‡РµРЅРёРµ СЃРїРёСЃРєР° РЅР°Р±РѕСЂР° РґР°РЅРЅС‹С…
def СЃРїРёСЃРѕРє_РґР°С‚Р°СЃРµС‚РѕРІ_LabStory():
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  get = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    datasets = json_response.json()
    meta = datasets['meta']
    get = 'http://labstory.neural-university.ru/api/my/datasets?per_page='+ str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    datasets = json_response.json()  
    i=1
    for d in datasets['data']:
      print('\033[1m', i, '. Р”Р°С‚Р°СЃРµС‚: ', d["name"], ' (\033[32mid ', d["id"], ')', ', \033[0m ', d["description"], sep='')
      print(' '*len(str(i)), ' РЎСЃС‹Р»РєР°: ', d['url'])
      print(' '*len(str(i)), ' -----------------------')
      print()
      i+=1
    print()
    print('\033[1m Р’СЃРµРіРѕ РґР°С‚Р°СЃРµС‚РѕРІ: ', meta['total'])
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)

  
# РџРѕР»СѓС‡РµРЅРёРµ СЃРїРёСЃРєР° РЅР°Р±РѕСЂР° РґР°РЅРЅС‹С…
def СЃРїРёСЃРѕРє_РґР°С‚Р°СЃРµС‚РѕРІ_LabStory():
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  get = 'http://labstory.neural-university.ru/api/my/datasets?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    datasets = json_response.json()
    meta = datasets['meta']
    get = 'http://labstory.neural-university.ru/api/my/datasets?per_page='+ str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    datasets = json_response.json()  
    i=1
    for d in datasets['data']:
      print('\033[1m', i, '. Р”Р°С‚Р°СЃРµС‚: ', d["name"], ' (\033[32mid ', d["id"], ')', ', \033[0m ', d["description"], sep='')
      print(' '*len(str(i)), ' РЎСЃС‹Р»РєР°: ', d['url'])
      print(' '*len(str(i)), ' -----------------------')
      print()
      i+=1
    print('\033[1m Р’СЃРµРіРѕ РґР°С‚Р°СЃРµС‚РѕРІ: ', meta['total'])
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
    
# РЈРґР°Р»РµРЅРёРµ РЅР°Р±РѕСЂР° РґР°РЅРЅС‹С…
def СѓРґР°Р»РёС‚СЊ_РґР°С‚Р°СЃРµС‚_LabStory(id):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  delete = f'http://labstory.neural-university.ru/api/my/datasets/{id}?access_token='+token
  json_response = requests.delete(delete)
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    print('Р”Р°С‚Р°СЃРµС‚ СѓСЃРїРµС€РЅРѕ СѓРґР°Р»РµРЅ')
  elif json_response.status_code == 500:
    print('РќРµРІРѕР·РјРѕР¶РЅРѕ СѓРґР°Р»РёС‚СЊ РґР°С‚Р°СЃРµС‚ РєРѕС‚РѕСЂС‹Р№ РїСЂРёРІСЏР·Р°РЅ Рє СЌРєСЃРїРµСЂРёРјРµРЅС‚Сѓ')
  elif json_response.status_code == 404:
    print('РќРµРІРѕР·РјРѕР¶РЅРѕ СѓРґР°Р»РёС‚СЊ РґР°С‚Р°СЃРµС‚. Р”Р°С‚Р°СЃРµС‚Р° СЃ С‚Р°РєРёРј id РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚...')
 
# Р’С‹Р±РѕСЂ РґР°С‚Р°СЃРµС‚Р° РїРѕ id
def РІС‹Р±СЂР°С‚СЊ_РґР°С‚Р°СЃРµС‚_LabStory(id):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  global РґР°С‚Р°СЃРµС‚_id
  get = f'http://labstory.neural-university.ru/api/my/datasets/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    datasets = json_response.json()
    РґР°С‚Р°СЃРµС‚_id = datasets['id']
    print('\033[1mР’С‹Р±СЂР°РЅ РґР°С‚Р°СЃРµС‚: ', )
    print(datasets["name"], ' (\033[32mid ', datasets["id"], ')', ', \033[0m ', datasets["description"], sep='')
    print('РЎСЃС‹Р»РєР°: ', datasets['url'])
  elif json_response.status_code == 404:
    print('\033[1mРћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('\033[0mР”Р°С‚Р°СЃРµС‚Р° СЃ id', id, '\033[0mРЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚ РІ Р’Р°С€РµРј Р°РєРєР°СѓРЅС‚Рµ')
    
# Р’С‹РІРѕРґ РёРЅС„РѕСЂРјР°С†РёРё Рѕ С‚РµРєСѓС‰РµРј РґР°С‚Р°СЃРµС‚Рµ
def С‚РµРєСѓС‰РёР№_РґР°С‚Р°СЃРµС‚():
  print('id', РґР°С‚Р°СЃРµС‚_id) 

# РЎРѕР·РґР°РЅРёРµ Р·Р°РґР°С‡Рё
def РґРѕР±Р°РІРёС‚СЊ_Р·Р°РґР°С‡Сѓ_LabStory(task_dict):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  global Р·Р°РґР°С‡Р°_id
  post = 'http://labstory.neural-university.ru/api/my/tasks?access_token='+token

  name = task_dict['name']
  description = task_dict['description']
  task = {
      "name": name,
      "description": description
  }
  json_response = requests.post(post, json=task, headers=headers)
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    print('Р—Р°РґР°С‡Р° СѓСЃРїРµС€РЅРѕ РґРѕР±Р°РІР»РµРЅР°')
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)

  # РїРѕР»СѓС‡Р°РµРј id С‚РѕР»СЊРєРѕ С‡С‚Рѕ РґРѕР±Р°РІР»РµРЅРЅРѕР№ Р·Р°РґР°С‡Рё С‡РµСЂРµР· РїРѕРёСЃРє id РІСЃРµС… Р·Р°РґР°С‡
  get = 'http://labstory.neural-university.ru/api/my/tasks?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    tasks = json_response.json()
    meta = tasks['meta']
    get = 'http://labstory.neural-university.ru/api/my/tasks?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    tasks = json_response.json()
    task_id = tasks['data'][-1]
    Р·Р°РґР°С‡Р°_id = task_id['id']
    print('id Р·Р°РґР°С‡Рё: ',task_id['id'] )
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
  
# РџРѕР»СѓС‡РµРЅРёРµ СЃРїРёСЃРєР° Р·Р°РґР°С‡
def СЃРїРёСЃРѕРє_Р·Р°РґР°С‡_LabStory():
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  get = 'http://labstory.neural-university.ru/api/my/tasks?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    tasks = json_response.json()
    meta = tasks['meta']
    i=1
    get = 'http://labstory.neural-university.ru/api/my/tasks?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    tasks = json_response.json()
    i=1
    for t in tasks['data']:
      print(f'\033[1m {i}. Р—Р°РґР°С‡Р°: {t["name"]} \033[32m(id {t["id"]})\033[0m, {t["description"]}')
      print(' '*len(str(i)), '  \033[1mР­РєСЃРїРµСЂРёРјРµРЅС‚РѕРІ:\033[0m ', t['experiments_count'])
      print('-----------------------')
      print()
      i+=1
    print()
    print('\033[1m Р’СЃРµРіРѕ Р·Р°РґР°С‡: ', meta['total'])
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
    
# Р’С‹Р±РѕСЂ Р·Р°РґР°С‡Рё РїРѕ id
def РІС‹Р±СЂР°С‚СЊ_Р·Р°РґР°С‡Сѓ_LabStory(id):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  global Р·Р°РґР°С‡Р°_id
  get = f'http://labstory.neural-university.ru/api/my/tasks/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    task = json_response.json()
    Р·Р°РґР°С‡Р°_id = task['id']
    print('\033[1mР’С‹Р±СЂР°РЅР° Р·Р°РґР°С‡Р° (id ', Р·Р°РґР°С‡Р°_id, '):', sep='')
    print(task['name'], ', \033[0m', task['description'], sep='')
    print('Р­РєСЃРїРµСЂРёРјРµРЅС‚РѕРІ: ', task['experiments_count'])
  elif json_response.status_code == 404:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print(f'Р—Р°РґР°С‡Рё c id {id} РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚ РІ Р’Р°С€РµРј Р°РєРєР°СѓРЅС‚Рµ.')
    
def С‚РµРєСѓС‰Р°СЏ_Р·Р°РґР°С‡Р°():
  print('id', Р·Р°РґР°С‡Р°_id)
  
# РЎРѕС…СЂР°РЅРµРЅРёРµ СЌРєСЃРїРµСЂРёРјРµРЅС‚Р°
def СЃРѕС…СЂР°РЅРёС‚СЊ_СЌРєСЃРїРµСЂРёРјРµРЅС‚_LabStory(experiment_dict):
  global model_architecture
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  if 'Р·Р°РґР°С‡Р°_id' not in globals():
    print('РќРµ РІС‹Р±СЂР°РЅР° Р·Р°РґР°С‡Р°')
    print('Р’С‹Р±РµСЂРёС‚Рµ РёР»Рё СЃРѕР·РґР°Р№С‚Рµ Р·Р°РґР°С‡Сѓ Рё РїРѕРІС‚РѕСЂРёС‚Рµ РґРµР№СЃС‚РІРёРµ')
    return
  if 'РґР°С‚Р°СЃРµС‚_id' not in globals():
    print('РќРµ РІС‹Р±СЂР°РЅ РґР°С‚Р°СЃРµС‚')
    print('Р’С‹Р±РµСЂРёС‚Рµ РёР»Рё СЃРѕР·РґР°Р№С‚Рµ РґР°С‚Р°СЃРµС‚ Рё РїРѕРІС‚РѕСЂРёС‚Рµ РґРµР№СЃС‚РІРёРµ')  
    return
  post = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token

  model_history_list = []
  for key in experiment_dict['history'].history.keys():
    model_history_list.append(experiment_dict['history'].history[f'{key}'])  

  loss_float, metric_float, val_loss_float, val_metric_float = '', '', '', ''
  loss, metric, val_loss, val_metric = '', '', '', ''

  if experiment_dict['loss'] == min:
    loss_float = min(model_history_list[0])
    loss = [round(v, 4) for v in model_history_list[0]]
    try:
      val_loss_float = min(model_history_list[2])
      val_loss = [round(v, 4) for v in model_history_list[2]]
    except IndexError:
      val_loss_float = 0
      val_loss = [0]
  elif experiment_dict['loss'] == max:
    loss_float = max(model_history_list[0])
    loss = [round(v, 4) for v in model_history_list[0]]
    try:
      val_loss_float = max(model_history_list[2])
      val_loss = [round(v, 4) for v in model_history_list[2]]
    except IndexError:
      val_loss_float = 0
      val_loss = [0]

  if experiment_dict['metrics'] == min:
    metric_float = min(model_history_list[1])
    metric = [round(v, 4) for v in model_history_list[1]]
    try:
      val_metric_float = min(model_history_list[3])
      val_metric = [round(v, 4) for v in model_history_list[3]]
    except IndexError:
      val_metric_float = 0
      val_metric = [0]
  elif experiment_dict['metrics'] == max:
    metric_float = max(model_history_list[1])
    metric = [round(v, 4) for v in model_history_list[1]]
    try:
      val_metric_float = max(model_history_list[3])
      val_metric = [round(v, 4) for v in model_history_list[3]]
    except IndexError:
      val_metric_float = 0
      val_metric = [0]
###############################
# РџРѕРёСЃРє Р°СЂС…РёС‚РµРєС‚СѓСЂС‹ РјРѕРґРµР»Рё    
###############################
  #sum_=0
  find = experiment_dict['function']
  #key_words = ['СЃР»РѕРё', 'РЎРІРµСЂС‚РѕС‡РЅС‹Р№2D', 'Dense', 'Conv2D']
  for i in reversed(experiment_dict['cache']):  
    #for j in range(len(key_words)):
      if (find in i) and ('experiment_dict' not in i):
        model_architecture = i
        break
      #else: sum_+=1
  #print(m_arc)
  #print(sum_, len(experiment_dict['cache']))   
###############################
  try:
    model_architecture
  except NameError:
    print('РќРµ РЅР°Р№РґРµРЅРѕ СЏС‡РµР№РєРё СЃ РєРѕРјРјРµРЅС‚Р°СЂРёРµРј СѓРєР°Р·Р°РЅРЅС‹Рј РІ РєР°С‡РµСЃС‚РІРµ Р·РЅР°С‡РµРЅРёСЏ РєР»СЋС‡Р° "function" РІ СЃР»РѕРІР°СЂРµ experiment_dict') 
    return
  experiment = {
    "task_id": Р·Р°РґР°С‡Р°_id,
    "user_id": user_id,
    "dataset_id": РґР°С‚Р°СЃРµС‚_id,
    "name_model": experiment_dict['name'],
    "loss": loss,
    "loss_float": loss_float,
    "val_loss": val_loss,
    "val_loss_float": val_loss_float,
    "metric": metric,
    "metric_float": metric_float,
    "val_metric": val_metric,
    "val_metric_float": val_metric_float,
    "function_creating_model": model_architecture,
    "function_data_preprocessing": experiment_dict['data_processing_type'],
    "description_model": experiment_dict['description'],
    "description_data_preprocessing": experiment_dict['description_data_processing_type'],
    "comment": experiment_dict['comment'],
    "tags": experiment_dict['tags'],
  }
  json_response = requests.post(post, json=experiment, headers=headers)
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
# РїРѕР»СѓС‡Р°РµРј id С‚РѕР»СЊРєРѕ С‡С‚Рѕ СЃРѕС…СЂР°РЅРµРЅРЅРѕРіРѕ СЌРєСЃРїРµСЂРёРјРµРЅС‚Р° РїРѕ id РІСЃРµС… СЌРєСЃРїРµСЂРёРјРµРЅС‚РѕРІ
    get = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token
    json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
    if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
      experiments = json_response.json()
      meta = experiments['meta']
      get = 'http://labstory.neural-university.ru/api/my/experiments?per_page=' + str(meta['total']) + '&access_token='+token
      json_response = requests.get(get, headers=headers)
      experiments = json_response.json()
      experiment_id = experiments['data'][-1]
      СЌРєСЃРїРµСЂРёРјРµРЅС‚_id = experiment_id['id']
      print('Р­РєСЃРїРµСЂРёРјРµРЅС‚ СѓСЃРїРµС€РЅРѕ СЃРѕС…СЂР°РЅРµРЅ')
      print('id СЌРєСЃРїРµСЂРёРјРµРЅС‚Р°:', experiment_id['id'] )
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
      

# РџРѕР»СѓС‡РµРЅРёРµ СЃРїРёСЃРєР° СЌРєСЃРїРµСЂРёРјРµС‚РѕРІ
def СЃРїРёСЃРѕРє_СЌРєСЃРїРµСЂРёРјРµС‚РѕРІ_LabStory():
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
  get = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    experiments = json_response.json()
    meta = experiments['meta']
    get = 'http://labstory.neural-university.ru/api/my/experiments?per_page=' + str(meta['total']) + '&access_token='+token
    json_response = requests.get(get, headers=headers)
    experiments = json_response.json()
    i=1
    print()
    for e in experiments['data']:
      print(i,'. id СЌРєСЃРїРµСЂРёРјРµРЅС‚Р°: ', e['id'], sep='')
      print('-----------------------')
      i+=1
    print()
    print('\033[1mР’СЃРµРіРѕ СЌРєСЃРїРµСЂРёРјРµРЅС‚РѕРІ: ', len(experiments['data']))
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print('РљРѕРґ РѕС€РёР±РєРё: ', json_response.status_code)
    print(json_response.text)
 
# РџРѕР»СѓС‡РµРЅРёРµ СЌРєСЃРїРµСЂРёРјРµС‚Р° РїРѕ id
def РїРѕСЃРјРѕС‚СЂРµС‚СЊ_СЌРєСЃРїРµСЂРёРјРµРЅС‚_РїРѕ_id_LabStory(id):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
    
  get = f'http://labstory.neural-university.ru/api/my/experiments/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    experiment = json_response.json()
    columns = ['РќР°Р·РІР°РЅРёРµ', 'РћРїРёСЃР°РЅРёРµ', 'val_metrics', 'val_loss',  'РљРѕРјРјРµРЅС‚Р°СЂРёР№', 'Р”Р°С‚Р°СЃРµС‚', 'Р”Р°С‚Р° СЃРѕР·РґР°РЅРёСЏ']
    data = [[experiment['name_model'],
             experiment['description_model'],
             experiment['val_metric_float'],
             experiment['val_loss_float'],
             experiment['comment'],
             experiment['dataset']['name'],
             experiment['created_at']]]     
    print(tabulate(data, headers=columns))
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print(f'Р­РєСЃРїРµСЂРёРјРµРЅС‚Р° c id {id} РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚ РІ Р’Р°С€РµРј Р°РєРєР°СѓРЅС‚Рµ.')
    
def РїРѕР»СѓС‡РёС‚СЊ_Р°СЂС…РёС‚РµРєС‚СѓСЂСѓ(id_exp):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')
    return
    
  get = f'http://labstory.neural-university.ru/api/my/experiments/{id_exp}?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    experiment = json_response.json()
    РїРѕР»СѓС‡РёС‚СЊ_Р°СЂС…РёС‚РµРєС‚СѓСЂСѓ = experiment['function_creating_model'] 
  else:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print(f'Р­РєСЃРїРµСЂРёРјРµРЅС‚Р° c id {id_exp} РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚ РІ Р’Р°С€РµРј Р°РєРєР°СѓРЅС‚Рµ.')
  return РїРѕР»СѓС‡РёС‚СЊ_Р°СЂС…РёС‚РµРєС‚СѓСЂСѓ

def РІСЃРµ_СЌРєСЃРїРµСЂРёРјРµРЅС‚С‹_РїРѕ_Р·Р°РґР°С‡Рµ(id):
  if 'token' not in globals():
    print('Р”Р»СЏ РІС‹РїРѕР»РЅРµРЅРёСЏ РґР°РЅРЅРѕР№ С„СѓРЅРєС†РёРё, РЅРµРѕР±С…РѕРґРёРјРѕ Р°РІС‚РѕСЂРёР·РёСЂРѕРІР°С‚СЊСЃСЏ.')

  get = f'http://labstory.neural-university.ru/api/my/tasks/{id}?access_token='+token
  json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
  if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
    get = 'http://labstory.neural-university.ru/api/my/experiments?access_token='+token
    json_response = requests.get(get, headers=headers) # РћС‚РїСЂР°РІР»СЏРµРј post-Р·Р°РїСЂРѕСЃ РЅР° СЃРµСЂРІРµСЂ
    if json_response.status_code == 200: # Р•СЃР»Рё РїСЂРёС€РµР» 200-С‹Р№ РєРѕРґ РѕС‚РІРµС‚Р° (РІСЃРµ С…РѕСЂРѕС€Рѕ)
      experiments = json_response.json()
      meta = experiments['meta']
      get = 'http://labstory.neural-university.ru/api/my/experiments?per_page=' + str(meta['total']) + '&access_token='+token
      json_response = requests.get(get, headers=headers)
      experiments = json_response.json()
      num_=1
      for i in range(len(experiments['data'])):
        if experiments['data'][i]['task_id'] == id:
          print(num_, '. id СЌРєСЃРїРµСЂРёРјРµРЅС‚Р°: ', experiments['data'][i]['id'], sep='')
          num_+=1
      print()
      print('-----------------------')
      print()
      print('\033[1mР’СЃРµРіРѕ СЌРєСЃРїРµСЂРёРјРµРЅС‚РѕРІ: ', num_-1)

  elif json_response.status_code == 404:
    print('РћС€РёР±РєР° РІС‹РїРѕР»РЅРµРЅРёСЏ Р·Р°РїСЂРѕСЃР°')
    print(f'Р—Р°РґР°С‡Рё c id {id} РЅРµ СЃСѓС‰РµСЃС‚РІСѓРµС‚ РІ Р’Р°С€РµРј Р°РєРєР°СѓРЅС‚Рµ.')


#Р”РµРјРѕРЅСЃС‚СЂР°С†РёСЏ

def РґРµРјРѕРЅСЃС‚СЂР°С†РёСЏ_РњРќРРЎРў():
  global x_train_org
  global y_train_org
  global x_test_org
  global y_test_org
  global x_train
  global x_train
  global x_test
  global x_test
  global model
  global imgs
  global weights
  global losses
  global accuracy
  global model
  global idx

  (x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()
  x_train = x_train_org.reshape((-1, 784))
  x_train = x_train/255

  x_test = x_test_org.reshape((-1, 784))
  x_test = x_test/255

  model = Sequential()
  model.add(Dense(10, use_bias=False, activation='softmax', input_dim=784))
  #model.trainable = False
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  imgs = []
  weights = [model.get_weights()[0]]
  losses = []
  accuracy = []
  idx = 0
  def on_batch_end(batch, logs):
    global idx
    if idx%60==0:
      imgs.append(model.get_weights()[0])
    idx+=1  

  def on_epoch_end(epoch, logs):
    weights.append(model.get_weights()[0])
    losses.append(logs['loss'])
    accuracy.append(logs['accuracy'])
  print('РћР¶РёРґР°Р№С‚Рµ, РёРґРµС‚ РїРѕРґРіРѕС‚РѕРІРєР° Рє РґРµРјРѕРЅСЃС‚СЂР°С†РёРё')
  lc = LambdaCallback(on_batch_end=on_batch_end, on_epoch_end=on_epoch_end)
  model.fit(x_train, y_train_org, batch_size=100, epochs=10, verbose=0, callbacks=[lc])
  display.clear_output(wait=True)
  print('Р”РµРјРѕРЅСЃС‚СЂР°С†РёСЏ РіРѕС‚РѕРІР°')
  
def РїРѕРєР°Р·Р°С‚СЊ_РёР·РјРµРЅРµРЅРёРµ_РјР°СЃРєРё_MNIST(number):
  for n in range(len(number)):
    numeric = np.array(imgs)[...,number[n]]
    f, ax = plt.subplots(3,8,figsize=(28, 10))
    for i in range(3):
      for j in range(8):
        ax[i,j].imshow(numeric[i*8+j].reshape((28,28)), cmap='gray')
        ax[i,j].axis('off')

       

def РїРѕРєР°Р·Р°С‚СЊ_РёР·РјРµРЅРµРЅРёРµ_РІРµСЃР°():
  num = 4
  global weights
  weights = np.array(weights)
  w = weights[:,np.random.randint(784),num]
  d = np.diff(w)
  import pandas as pd
  df = pd.DataFrame(columns=['Р—РЅР°С‡РµРЅРёРµ РІРµСЃР°','РР·РјРµРЅРµРЅРёРµ РІРµСЃР°','РќРѕРІРѕРµ Р·РЅР°С‡РµРЅРёРµ', 'РћС€РёР±РєР° РјРѕРґРµР»Рё', 'РўРѕС‡РЅРѕСЃС‚СЊ РјРѕРґРµР»Рё'])
  df.index.name = 'Р­РїРѕС…Р°'
  for i in range(10):
    df.loc[i] = [w[:-1][i], d[i], w[1:][i], losses[i], accuracy[i]]
  return df.head(10)

def РґРµРјРѕРЅСЃС‚СЂР°С†РёСЏ_РђР’РўРћ():
  global idx
  global imgs
  global РѕР±СѓС‡Р°СЋС‰Р°СЏ_РІС‹Р±РѕСЂРєР°
  global part1
  global part2
  global part3
  global part4

  РїСѓС‚СЊ = 'Р°РІС‚РѕРјРѕР±РёР»Рё'
  РєРѕСЌС„_СЂР°Р·РґРµР»РµРЅРёСЏ=0.9
  РѕР±СѓС‡Р°СЋС‰Р°СЏ_РІС‹Р±РѕСЂРєР° = [] # РЎРѕР·РґР°РµРј РїСѓСЃС‚РѕР№ СЃРїРёСЃРѕРє, РІ РєРѕС‚РѕСЂС‹Р№ Р±СѓРґРµРј СЃРѕР±РёСЂР°С‚СЊ РїСЂРёРјРµСЂС‹ РёР·РѕР±СЂР°Р¶РµРЅРёР№ РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРё
  y_train = [] # РЎРѕР·РґР°РµРј РїСѓСЃС‚РѕР№ СЃРїРёСЃРѕРє, РІ РєРѕС‚РѕСЂС‹Р№ Р±СѓРґРµРј СЃРѕР±РёСЂР°С‚СЊ РїСЂР°РІРёР»СЊРЅС‹Рµ РѕС‚РІРµС‚С‹ (РјРµС‚РєРё РєР»Р°СЃСЃРѕРІ: 0 - Р¤РµСЂСЂР°СЂРё, 1 - РњРµСЂСЃРµРґРµСЃ, 2 - Р РµРЅРѕ)
  x_test = [] # РЎРѕР·РґР°РµРј РїСѓСЃС‚РѕР№ СЃРїРёСЃРѕРє, РІ РєРѕС‚РѕСЂС‹Р№ Р±СѓРґРµРј СЃРѕР±РёСЂР°С‚СЊ РїСЂРёРјРµСЂС‹ РёР·РѕР±СЂР°Р¶РµРЅРёР№ С‚РµСЃС‚РѕРІРѕР№ РІС‹Р±РѕСЂРєРё
  y_test = [] # РЎРѕР·РґР°РµРј РїСѓСЃС‚РѕР№ СЃРїРёСЃРѕРє, РІ РєРѕС‚РѕСЂС‹Р№ Р±СѓРґРµРј СЃРѕР±РёСЂР°С‚СЊ РїСЂР°РІРёР»СЊРЅС‹Рµ РѕС‚РІРµС‚С‹ (РјРµС‚РєРё РєР»Р°СЃСЃРѕРІ: 0 - Р¤РµСЂСЂР°СЂРё, 1 - РњРµСЂСЃРµРґРµСЃ, 2 - Р РµРЅРѕ)

  for j, d in enumerate(sorted(os.listdir(РїСѓС‚СЊ))):
    files = sorted(os.listdir(РїСѓС‚СЊ + '/'+d))    
    count = len(files) * РєРѕСЌС„_СЂР°Р·РґРµР»РµРЅРёСЏ
    for i in range(len(files)):
      sample = image.load_img(РїСѓС‚СЊ + '/' +d +'/'+files[i], target_size=(54, 96)) # Р—Р°РіСЂСѓР¶Р°РµРј РєР°СЂС‚РёРЅРєСѓ
      img_numpy = np.array(sample) # РџСЂРµРѕР±СЂР°Р·СѓРµРј Р·РѕР±СЂР°Р¶РµРЅРёРµ РІ numpy-РјР°СЃСЃРёРІ
      if i<count:
        РѕР±СѓС‡Р°СЋС‰Р°СЏ_РІС‹Р±РѕСЂРєР°.append(img_numpy) # Р”РѕР±Р°РІР»СЏРµРј РІ СЃРїРёСЃРѕРє x_train СЃС„РѕСЂРјРёСЂРѕРІР°РЅРЅС‹Рµ РґР°РЅРЅС‹Рµ
        y_train.append(j) # Р”РѕР±Р°РІР»РµСЏРј РІ СЃРїРёСЃРѕРє y_train Р·РЅР°С‡РµРЅРёРµ 0-РіРѕ РєР»Р°СЃСЃР°
      else:
        x_test.append(img_numpy) # Р”РѕР±Р°РІР»СЏРµРј РІ СЃРїРёСЃРѕРє x_test СЃС„РѕСЂРјРёСЂРѕРІР°РЅРЅС‹Рµ РґР°РЅРЅС‹Рµ
        y_test.append(j) # Р”РѕР±Р°РІР»РµСЏРј РІ СЃРїРёСЃРѕРє y_test Р·РЅР°С‡РµРЅРёРµ 0-РіРѕ РєР»Р°СЃСЃР°
  display.clear_output(wait=True)
  x_train = np.array(РѕР±СѓС‡Р°СЋС‰Р°СЏ_РІС‹Р±РѕСЂРєР°) # РџСЂРµРѕР±СЂР°Р·СѓРµРј Рє numpy-РјР°СЃСЃРёРІСѓ
  y_train = np.array(y_train) # РџСЂРµРѕР±СЂР°Р·СѓРµРј Рє numpy-РјР°СЃСЃРёРІСѓ
  x_test = np.array(x_test) # РџСЂРµРѕР±СЂР°Р·СѓРµРј Рє numpy-РјР°СЃСЃРёРІСѓ
  y_test = np.array(y_test) # РџСЂРµРѕР±СЂР°Р·СѓРµРј Рє numpy-РјР°СЃСЃРёРІСѓ
  x_train = x_train/255.
  x_test = x_test/255.
  
  inp = Input(shape=(54,96,3))
  x1 = Conv2D(8, (3,3), activation='relu', padding='same') (inp)
  x2 = Conv2D(8, (3,3), activation='relu', padding='same') (x1)
  x3 = Conv2D(8, (3,3), activation='relu', padding='same') (x2)
  x4 = Conv2D(8, (3,3), activation='relu', padding='same') (x3)
  x = Flatten() (x4)
  x = Dense(10, activation='softmax')(x)

  part1 = Model(inp, x1)
  part2 = Model(inp, x2)
  part3 = Model(inp, x3)
  part4 = Model(inp, x4)
  model = Model(inp, x)
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  imgs = []
  sample = random.choice(x_train)
  idx = 0
  def on_batch_end(batch, logs):
    global idx
    if idx%25==0:
      imgs.append(part1.predict(sample[None,...]).reshape((54,96,8)))
    idx+=1
  print('РћР¶РёРґР°Р№С‚Рµ, РёРґРµС‚ РїРѕРґРіРѕС‚РѕРІРєР° Рє РґРµРјРѕРЅСЃС‚СЂР°С†РёРё')
  lc = LambdaCallback(on_batch_end=on_batch_end)
  model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=[lc], verbose=0)
  display.clear_output(wait=True)
  print('Р”РµРјРѕРЅСЃС‚СЂР°С†РёСЏ РіРѕС‚РѕРІР°')
  
def РїРѕРєР°Р·Р°С‚СЊ_РёР·РјРµРЅРµРЅРёРµ_РјР°СЃРєРё_РђР’РўРћ(num):
  f, ax = plt.subplots(3,6,figsize=(20, 8))
  for i in range(3):
    for j in range(6):
      ax[i,j].imshow(imgs[i*6+j][:,:,num].reshape((54,96)), cmap='gray')
      ax[i,j].axis('off')

def РїРѕРєР°Р·Р°С‚СЊ_РјР°СЃРєРё():
  import warnings
  warnings.filterwarnings('ignore')
  sample = random.choice(РѕР±СѓС‡Р°СЋС‰Р°СЏ_РІС‹Р±РѕСЂРєР°)
  images1 = part1.predict(sample[None,...]).reshape((54,96,8))
  images2 = part2.predict(sample[None,...]).reshape((54,96,8))
  images3 = part3.predict(sample[None,...]).reshape((54,96,8))
  images4 = part4.predict(sample[None,...]).reshape((54,96,8))
  print('*** РћСЂРёРіРёРЅР°Р»СЊРЅРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёРµ ***')
  plt.imshow(sample)
  plt.axis('off')
  plt.show()
  print()

  print('*** РљР°СЂС‚С‹ РїРµСЂРІРѕРіРѕ СЃРІРµСЂС‚РѕС‡РЅРѕРіРѕ СЃР»РѕСЏ ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images1[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
  print()

  print('*** РљР°СЂС‚С‹ РІС‚РѕСЂРѕРіРѕ СЃРІРµСЂС‚РѕС‡РЅРѕРіРѕ СЃР»РѕСЏ ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images2[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
  print()

  print('*** РљР°СЂС‚С‹ С‚СЂРµС‚СЊРµРіРѕ СЃРІРµСЂС‚РѕС‡РЅРѕРіРѕ СЃР»РѕСЏ ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images3[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
  print()

  print('*** РљР°СЂС‚С‹ С‡РµС‚РІРµСЂС‚РѕРіРѕ СЃРІРµСЂС‚РѕС‡РЅРѕРіРѕ СЃР»РѕСЏ ***')
  f, ax = plt.subplots(1,8, figsize=(40,25))
  for i in range(8):
    ax[i].imshow(images4[:,:,i], cmap='gray')
    ax[i].axis('off')
  plt.show()
