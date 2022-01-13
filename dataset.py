from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
import cv2
import time


class CustomDataset(Dataset):
    def __init__(self, args, df, transforms=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.num_classes = args.num_classes


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.args.dataset_dir, self.df["path"][idx])
        # img_path = img_path.replace("/", "\\")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']

        label = self.df["label"][idx]

        return img, label



class TestDataset(Dataset):
    def __init__(self, args, df, transforms=None):
        super().__init__()
        self.args = args
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.num_classes = args.num_classes


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.args.dataset_dir, self.df["path"][idx])
        # img_path = img_path.replace("/", "\\")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']

        label = self.df["label"][idx]

        return img, label, img_path

