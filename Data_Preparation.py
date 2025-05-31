import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
import kagglehub

# Download latest version
# DATASET_PATH = kagglehub.dataset_download("satishpaladi11/mechanic-component-images-normal-defected")
# print(f"DATASET_PATH: {DATASET_PATH}")

IMG_SIZE = (224, 224)

# 1. Image Resizer + Standardization/Normalizer + Image transform + Tensor transform
def resize_trans(IMG_SIZE):
    return A.Compose([
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet нормализация
        ToTensorV2()
    ])

# 2. Data Augmentation + Image transform + Tensor transform
def data_augmentation(IMG_SIZE):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet нормализация
        ToTensorV2()
    ])


class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image) / 255.0  # Нормализуем в [0, 1]

        if self.augment:
            image = self.augment(image=image)['image']
        elif self.transform:
            image = self.transform(image=image)['image']

        # Отладочный вывод
        # print(f"Image {img_path} min: {image.min().item()}, max: {image.max().item()}")
        return image, label

def visualize_random_images(dataset, class_names):
    class_indices = {i: [] for i in range(len(class_names))}
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)

    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 5))
    for class_id, indices in class_indices.items():
        if indices:
            random_idx = random.choice(indices)
            img, label = dataset[random_idx]
            img = img.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            axes[class_id].imshow(img)
            axes[class_id].set_title(f"Class: {class_names[label]}")
            axes[class_id].axis('off')
    plt.show()

# def visualize_augmented_images(dataset, class_names, num_samples=5):
#     fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
#     random_indices = random.sample(range(len(dataset)), num_samples)
#     for i, idx in enumerate(random_indices):
#         img, label = dataset[idx]
#         img = img.permute(1, 2, 0).numpy()
#         img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.406])
#         img = np.clip(img, 0, 1)
#         axes[i].imshow(img)
#         axes[i].set_title(f"Class: {class_names[label]}")
#         axes[i].axis('off')
#     plt.show()


# 6. Index Splitter + Handle Imbalanced Classes + Training set + Validation set
def prepare_data(dataset_path):
    resize_transform = resize_trans(IMG_SIZE)
    augmentation_transform = data_augmentation(IMG_SIZE)

    dataset = datasets.ImageFolder(dataset_path)
    image_paths = [os.path.join(dataset_path, img[0]) for img in dataset.imgs]
    labels = [img[1] for img in dataset.imgs]
    class_names = dataset.classes
    train_idx, val_idx = train_test_split(
        range(len(image_paths)),
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    print("Sample train paths:", train_paths[:5])
    print("Sample val paths:", val_paths[:5])
    class_counts = Counter(train_labels)
    print("Class distribution in training set:", class_counts)
    print("Class distribution in validation set:", Counter(val_labels))
    num_samples = len(train_labels)
    class_weights = {i: num_samples / (len(class_names) * count) for i, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataset = DefectDataset(train_paths, train_labels, transform=None, augment=augmentation_transform)
    val_dataset = DefectDataset(val_paths, val_labels, transform=resize_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_loader, val_loader, class_names

# Код для тестирования модуля (опционально)
# if __name__ == '__main__':
#     # Загрузка датасета
#     DATASET_PATH = kagglehub.dataset_download("satishpaladi11/mechanic-component-images-normal-defected")
#     train_loader, val_loader, class_names = prepare_data(DATASET_PATH)
#     print(f"Classes: {class_names}")
#     print(f"Training set size: {len(train_loader.dataset)}")
#     print(f"Validation set size: {len(val_loader.dataset)}")
#     for images, labels in train_loader:
#         print(f"Batch shape: {images.shape}, Labels: {labels}")
#         break
