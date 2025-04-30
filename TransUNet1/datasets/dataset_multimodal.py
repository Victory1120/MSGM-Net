import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2




def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    image = np.flip(image, axis=0).copy()
    label = np.flip(label, axis=0).copy()

    image = np.flip(image, axis=1).copy()
    label = np.flip(label, axis=1).copy()

    return image, label


def random_rotate(image, label):
    angle = round(np.random.randint(-10, 10), 2)
    image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
    label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        print(np.unique(label))

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, z = image.shape


        if x != self.output_size[0] or y != self.output_size[1]:
            # for i in range(z):
            #     image1 = image[:,:,i]
            #     image1 = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?  240--> 224
            #     image2 =
            image1 = image[:,:,0]
            image1 = zoom(image1, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?  240--> 224
            image2 = image[:, :, 1]
            image2 = zoom(image2, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?  240--> 224
            image3 = image[:, :, 2]
            image3 = zoom(image3, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?  240--> 224
            image4 = image[:, :, 3]
            image4 = zoom(image4, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?  240--> 224
            image = np.stack([image1, image2, image3, image4], axis=-1)  # (224,224,4)
            label = label.astype(np.int32)
            label = cv2.resize(
                label,
                (self.output_size[1], self.output_size[0]),  # (width, height)
                interpolation=cv2.INTER_NEAREST
            )
            # label = label.astype(np.int32)  # 确保输入是整数
            # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0,
            #              output=np.int32)  # 强制输出为整数)  # 矩阵缩放，而不是随机裁剪 (224,224)
            print(np.unique(label))
            # print(image)

            # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 矩阵缩放，而不是随机裁剪 (224,224)
            # print(image)
        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.float32))
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        # label = np.ascontiguousarray(label)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int32))  # 直接转int64
        # label = label.astype(np.int32)  # 二次保险
        # label = torch.from_numpy(label.astype(np.int64))  # 直接转int64
        # # label = torch.from_numpy(label.astype(np.int32))
        # label = torch.from_numpy(label.astype(np.int32))
        # print(np.max(label.numpy()))  # 如果需要打印 NumPy 数组的最大值
        # print(image)
        print(np.unique(label))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']  # image为224，224，4 label为224,224
            # 输出唯一值
            unique_values = np.unique(label)
            print("唯一值：")
            print(unique_values)

            # 输出最大值
            max_value = np.max(label)
            print("最大值：")
            print(max_value)
            # print(image.shape) 这时候还是512，512
        #
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npz.h5".format(vol_name)
            data = h5py.File(filepath)
            image1, label = data['image1'][:], data['label'][:]
            image2 = data['image2'][:]
            image3 = data['image3'][:]
            image4 = data['image4'][:]
            image1 = torch.from_numpy(image1.astype(np.float32))
            image2 = torch.from_numpy(image2.astype(np.float32))
            image3 = torch.from_numpy(image3.astype(np.float32))
            image4 = torch.from_numpy(image4.astype(np.float32))
            # image = image.permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))

        sample = {'image1': image1,'image2': image2,'image3': image3,'image4': image4, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
