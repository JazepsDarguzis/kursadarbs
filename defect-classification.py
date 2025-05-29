#%%
# Требуемые пакеты:
# pip install numpy pandas matplotlib seaborn opencv-python kagglehub tensorflow scikit-learn
# Для keras:
# pip install keras
# Для работы с изображениями (Pillow):
# pip install pillow
#tensorflow.keras ошибки могут возникнуть из-за питона 3.13, используй 3.10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
import kagglehub
#%%
base_path = kagglehub.dataset_download("satishpaladi11/mechanic-component-images-normal-defected")
#%% md
# ### тут можно посмотреть, что в датасете
#%%
print(f"There are {len(os.listdir(base_path))} type of dataset")
#%%
data_set = []
amount_of_each = []
for folder in os.listdir(base_path):
    data_set.append(folder)
    amount_of_each.append(len(os.listdir(os.path.join(base_path,folder))))
    
plt.figure(figsize=(9,6))
sns.set_style('darkgrid')
sns.barplot(x = data_set,y = amount_of_each)
#%% md
# ## визуализация изображений
#%%
fig,ax = plt.subplots(3,3,figsize=(9,8))
for i in range(3):
    file_name = random.choice(os.listdir(os.path.join(base_path,data_set[i])))
    folder = os.path.join(base_path,data_set[i])
    for j in range(3):
        img = cv2.imread(os.path.join(folder,file_name))
        ax[i,j].imshow(img)
#%%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,ResNet50,EfficientNetB6
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
tf.config.list_physical_devices('GPU') 
#%% md
# ## модель на претрене ResNet50
#%%
resnet = ResNet50(include_top=False,input_shape=(80,80,3))
for layer in resnet.layers:
    layer.trainable = False
flat = Flatten()((resnet.layers[-1].output))
dense1 = Dense(1024,activation='relu')(flat)
dense2 = Dense(512,activation='relu')(dense1)
drop = Dropout(0.2)(dense2)
model_output = Dense(3,activation = 'softmax')(drop)
model = Model(resnet.input,model_output)
# model.summary()
#%%
callback = ModelCheckpoint('./checkpoint.weights.h5',save_weights_only=True,
    monitor='val_accuracy',
    mode='max',)
earlystop = EarlyStopping(monitor='val_accuracy',patience=15,mode='max',restore_best_weights=True)
model.compile(optimizer=RMSprop(learning_rate=0.001),loss = 'categorical_crossentropy',metrics=['accuracy'])
#%%
data_gen = ImageDataGenerator(brightness_range=[1.5,2.5],
                              rotation_range = 0.6,validation_split=0.3)
#%%
train_data = data_gen.flow_from_directory(base_path,target_size=(80,80),subset='training')
valid_data = data_gen.flow_from_directory(base_path,target_size=(80,80),subset='validation')
#%%
history= model.fit(train_data,epochs=50,validation_data=valid_data,callbacks=[callback])
#меняй эпохи, если не хватает памяти
#%%
model.save("model_for_raw_material.keras")
# сменил на керас, чтобы не было варнинга
#%%

#там будет рекомендации в терминале, что делать с варнингом: это предупреждение связано с внутренней
# реализацией pydataset и не является ошибкой. Оно не влияет на работу кода, обучению и сохранению модели