from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
from classification.utils_cls import *


class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs, self.scale, self.ratio)
        imgs = transforms.functional.resized_crop(
            imgs, i, j, h, w, self.size, self.interpolation)
        return imgs


class ClsLoader(Dataset):
    def __init__(self, path_image, csv, transform=True, typeData="train", sup_loss=config["SUP_LOSS"]):
        self.transform = transform if typeData == "train" else False  # augment data bool
        self.path_image = path_image
        self.ids = csv["image_id"]
        self.label = csv["abnormal"]
        self.sup_loss = sup_loss

    def __len__(self):
        return len(self.ids)

    def rotate(self, image, degrees=(-30, 30), p=0.2):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
        return image

    def horizontal_flip(self, image, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def vertical_flip(self, image, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def random_resized_crop(self, image, p=0.5):
        if torch.rand(1) < p:
            image = RandomCrop(config["SIZE_IMAGE_CLS"],
                               scale=(0.7, 0.9))(image)
        return image

    def augment(self, image):
        # image = self.rotate(image)
        image = self.horizontal_flip(image)
        image = self.vertical_flip(image)
        image = self.random_resized_crop(image)
        return image

    def __getitem__(self, idx):
        image = Image.open(f"{self.path_image}{self.ids[idx]}.png")
        label = self.label[idx]
    ####################### augmentation data ##############################
        if self.sup_loss:
            image = self.augment(image)
            image_1 = self.augment(image)
        elif self.transform:
            image = self.augment(image)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        label = torch.as_tensor([label], dtype=torch.int64)

        if self.sup_loss:
            image_1 = transforms.ToTensor()(image_1)
            image_1 = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_1)

            return image, image_1, label
        return image, label
