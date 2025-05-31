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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from Data_Preparation import prepare_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback


# Функция для преобразования DataLoader в генератор Keras
def dataloader_to_keras_generator(dataloader, num_classes):
    while True:
        for images, labels in dataloader:
            images_np = images.numpy()
            images_np = np.transpose(images_np, (0, 2, 3, 1))  # (batch_size, H, W, C)
            labels_np = labels.numpy()
            labels_one_hot = tf.keras.utils.to_categorical(labels_np, num_classes=num_classes)
            # print(f"Image batch shape: {images_np.shape}, min: {images_np.min()}, max: {images_np.max()}")
            yield images_np, labels_one_hot


class TrainingProgressCallback(Callback):
    def __init__(self, progress_bar, text_output, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.text_output = text_output
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        percent = int((epoch / self.total_epochs) * 100)
        self.progress_bar.setValue(percent)
        self.text_output.appendPlainText(f"▶ Epoch {epoch+1}/{self.total_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy")
        loss = logs.get("loss")
        val_acc = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")

        self.text_output.appendPlainText(
            f"✅ Epoch {epoch+1}: "
            f"acc={acc:.4f}, loss={loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}"
        )

    def on_train_end(self, logs=None):
        self.progress_bar.setValue(100)
        self.text_output.appendPlainText("🎉 Обучение завершено!")

def run_training(progress_bar=None, text_output=None):
    tf.config.list_physical_devices('GPU')

    #%%
    base_path = kagglehub.dataset_download("satishpaladi11/mechanic-component-images-normal-defected")
    train_loader, val_loader, class_names = prepare_data(base_path)
    num_classes = len(class_names)
    print(f"Classes: {class_names}, Number of classes: {num_classes}")
    #%% mds
    # ### тут можно посмотреть, что в датасете
    #%%
    # print(f"There are {len(os.listdir(base_path))} type of dataset")
    #%%
    data_set = []
    amount_of_each = []
    for folder in os.listdir(base_path):
        data_set.append(folder)
        amount_of_each.append(len(os.listdir(os.path.join(base_path, folder))))
    plt.figure(figsize=(9, 6))
    sns.set_style('darkgrid')
    sns.barplot(x=data_set, y=amount_of_each)
    plt.title('Number of Images per Class')
    # plt.show()

    #%% md
    # ## визуализация изображений
    #%%
    fig, ax = plt.subplots(3, 3, figsize=(9, 8))
    for i in range(min(3, len(data_set))):
        file_name = random.choice(os.listdir(os.path.join(base_path, data_set[i])))
        folder = os.path.join(base_path, data_set[i])
        for j in range(3):
            img = cv2.imread(os.path.join(folder, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i, j].imshow(img)
            ax[i, j].set_title(f"Class: {data_set[i]}")
            ax[i, j].axis('off')
    # plt.show()
    #%%

    # ## модель на претрене ResNet50
    #%%
    train_generator = dataloader_to_keras_generator(train_loader, num_classes)
    val_generator = dataloader_to_keras_generator(val_loader, num_classes)

    # Проверка одного батча
    sample_images, sample_labels = next(train_generator)
    print(f"Sample train batch shape: {sample_images.shape}, min: {sample_images.min()}, max: {sample_images.max()}")
    print(f"Sample train labels shape: {sample_labels.shape}")

    # Модель ResNet50
    resnet = ResNet50(include_top=False, input_shape=(224, 224, 3))
    for layer in resnet.layers:
        layer.trainable = False
    flat = Flatten()(resnet.layers[-1].output)
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(flat)
    drop = Dropout(0.5)(dense1)
    model_output = Dense(num_classes, activation='softmax')(drop)
    model = Model(resnet.input, model_output)
    # model.summary()
    #%%
    # Компиляция модели
    callback = ModelCheckpoint('./checkpoint.weights.h5', save_weights_only=True,
                               monitor='val_accuracy', mode='max')
    earlystop = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    #%%
    # Обучение модели
    steps_per_epoch = len(train_loader.dataset) // train_loader.batch_size
    validation_steps = len(val_loader.dataset) // val_loader.batch_size
    progress_cb = TrainingProgressCallback(progress_bar, text_output, total_epochs=5)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=5,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[callback, earlystop, progress_cb]
    )
    #меняй эпохи, если не хватает памяти

    # Визуализация результатов обучения
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    # plt.show()

    model.save("model_for_raw_material.keras")
    # сменил на керас, чтобы не было варнинга
    return history, base_path, data_set
    #там будет рекомендации в терминале, что делать с варнингом: это предупреждение связано с внутренней
    # реализацией pydataset и не является ошибкой. Оно не влияет на работу кода, обучению и сохранению модели