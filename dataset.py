import os
import random
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LensDataset(Dataset):

    def __init__(self, path: str, train_flag: bool=True, img_size: int=512, transformer=None) -> None:
        self._path = path
        self._img_size = img_size
        if transformer == None:
            self._transformer = transforms.ToTensor()
        else:
            self._transformer = transformer     
        
        if train_flag:
            self._data_sub_dir = 'train_data'
        else:
            self._data_sub_dir = 'test_data'

        self._data_name_list = os.listdir(os.path.join(self._path, self._data_sub_dir))
        random.shuffle(self._data_name_list) 

    def __len__(self) -> int:
        return len(self._data_name_list)

    def __getitem__(self, index: int):
        target_name = self._data_name_list[index]
        img_path = os.path.join(self._path, self._data_sub_dir, target_name)
        label_path = os.path.join(self._path, 'train_label', target_name)

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)

        img = cv2.resize(img, (self._img_size, self._img_size), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (self._img_size, self._img_size), interpolation=cv2.INTER_AREA)[:, :, 1]
        gt = label.copy()

        img = self._transformer(img)
        label = self._transformer(label)

        img = torch.unsqueeze(img, 0)

        img.type(torch.FloatTensor)
        label.type(torch.LongTensor)

        return img, label, gt, target_name


if __name__ == '__main__':

    pass

