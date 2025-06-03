#%%
# Требуемые пакеты:
# pip install numpy pandas matplotlib seaborn opencv-python kagglehub tensorflow scikit-learn
# Для keras:
# pip install keras
# Для работы с изображениями (Pillow):
# pip install pillow
#tensorflow.keras ошибки могут возникнуть из-за питона 3.13, используй 3.10

import numpy as np
import random
from PIL import Image
from torchvision import datasets
import kagglehub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from Data_Preparation import prepare_data, data_augmentation, DefectDataset
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix


class TrainingProgressCallback(Callback):
    def __init__(self, progress_bar, text_output, text_signal, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.text_output = text_output
        self.text_signal = text_signal
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        percent = int((epoch / self.total_epochs) * 100)
        self.progress_bar.setValue(percent)
        self.text_signal.emit(f"▶ Epoch {epoch+1}/{self.total_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy")
        loss = logs.get("loss")
        val_acc = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")
        self.text_signal.emit(
            f"✅ Epoch {epoch+1}: "
            f"accuracy={acc:.4f}, losses={loss:.4f}, "
            f"accuracy_validation={val_acc:.4f}, loss_validation={val_loss:.4f}"
        )

    def on_train_end(self, logs=None):
        self.progress_bar.setValue(100)
        self.text_signal.emit("🎉 The training is complete!")
        self.text_signal.emit("📊 The statistics are ready!")


def run_training(progress_bar=None, text_output=None, text_signal=None):
    tf.config.list_physical_devices('GPU')
    base_path = kagglehub.dataset_download("satishpaladi11/mechanic-component-images-normal-defected")
    train_loader, val_loader, class_names = prepare_data(base_path)
    num_classes = len(class_names)

    # Собираем 5 случайных изображений (по одному на класс, если возможно)
    sample_images = []
    dataset = datasets.ImageFolder(base_path)
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (img_path, label) in enumerate(dataset.imgs):
        class_indices[label].append(img_path)
    for class_id in range(min(5, num_classes)):
        if class_indices[class_id]:
            img_path = random.choice(class_indices[class_id])
            sample_images.append((img_path, class_names[class_id]))

    # Собираем аугментированные версии тех же изображений
    augmentation_transform = data_augmentation((224, 224))
    augmented_images = []
    for img_path, class_name in sample_images:
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        aug_image = augmentation_transform(image=image)['image']
        augmented_images.append((aug_image, class_name))

    # print(f"Classes: {class_names}, Number of classes: {num_classes}")
    #%% mds
    # ### тут можно посмотреть, что в датасете
    #%%
    # print(f"There are {len(os.listdir(base_path))} type of dataset")
    #%%
    # data_set = []
    # amount_of_each = []
    # for folder in os.listdir(base_path):
    #     data_set.append(folder)
    #     amount_of_each.append(len(os.listdir(os.path.join(base_path, folder))))
    # plt.figure(figsize=(9, 6))
    # sns.set_style('darkgrid')
    # sns.barplot(x=data_set, y=amount_of_each)
    # plt.title('Number of Images per Class')
    # plt.show()

    #%% md
    # ## визуализация изображений
    #%%
    # fig, ax = plt.subplots(3, 3, figsize=(9, 8))
    # for i in range(min(3, len(data_set))):
    #     file_name = random.choice(os.listdir(os.path.join(base_path, data_set[i])))
    #     folder = os.path.join(base_path, data_set[i])
    #     for j in range(3):
    #         img = cv2.imread(os.path.join(folder, file_name))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         ax[i, j].imshow(img)
    #         ax[i, j].set_title(f"Class: {data_set[i]}")
    #         ax[i, j].axis('off')
    # plt.show()
    #%%

    # ## модель на претрене ResNet50
    #%%
    train_generator = dataloader_to_keras_generator(train_loader, num_classes)
    val_generator = dataloader_to_keras_generator(val_loader, num_classes)

    # Проверка одного батча
    # sample_images, sample_labels = next(train_generator)
    # print(f"Sample train batch shape: {sample_images.shape}, min: {sample_images.min()}, max: {sample_images.max()}")
    # print(f"Sample train labels shape: {sample_labels.shape}")

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

    # Обучение модели
    steps_per_epoch = len(train_loader.dataset) // train_loader.batch_size
    validation_steps = len(val_loader.dataset) // val_loader.batch_size
    progress_cb = TrainingProgressCallback(progress_bar, text_output, text_signal, total_epochs=50)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[callback, earlystop, progress_cb]
    )
    #меняй эпохи, если не хватает памяти

    # Визуализация результатов обучения
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='Train Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    # plt.title('Accuracy over Epochs')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Val Loss')
    # plt.title('Loss over Epochs')
    # plt.legend()
    # plt.show()

    # Вычисление матрицы ошибок
    val_images, val_labels = [], []
    for images, labels in val_loader:
        val_images.append(images.numpy())
        val_labels.append(labels.numpy())
    val_images = np.concatenate(val_images)
    val_labels = np.concatenate(val_labels)
    val_images = np.transpose(val_images, (0, 2, 3, 1))
    predictions = model.predict(val_images)
    predicted_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(val_labels, predicted_labels)
    cm_data = (cm, class_names)

    model.save("model_for_raw_material.keras")
    # сменил на керас, чтобы не было варнинга
    return history, base_path, class_names, sample_images, augmented_images, cm_data
    #там будет рекомендации в терминале, что делать с варнингом: это предупреждение связано с внутренней
    # реализацией pydataset и не является ошибкой. Оно не влияет на работу кода, обучению и сохранению
    #
    # Функция для преобразования DataLoader в генератор Keras
def dataloader_to_keras_generator(dataloader, num_classes):
    while True:
        for images, labels in dataloader:
            images_np = images.numpy()
            images_np = np.transpose(images_np, (0, 2, 3, 1))
            labels_np = labels.numpy()
            labels_one_hot = tf.keras.utils.to_categorical(labels_np, num_classes=num_classes)
            yield images_np, labels_one_hot