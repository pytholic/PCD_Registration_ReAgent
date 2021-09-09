#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from torch.utils.data import Dataset
import torchvision
import os
import h5py
import pickle  # TODO or use h5py instead?
import trimesh
import open3d as o3d
import glob

import config as cfg
import dataset.augmentation as Transforms


# In[28]:


class CustomDataset(Dataset):
    
    def __init__(self, split, noise_type):
        dataset_path = cfg.CUSTOM_PATH

        self.samples, self.labels = self.get_samples(dataset_path, split)
        self.transforms = self.get_transforms(split, noise_type)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'points': self.samples[item, :, :], 'label': self.labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "clean":
            # 1-1 correspondence for each point (resample first before splitting), no noise
            if split == "train":
                transforms = [Transforms.Resampler(2048),
                              Transforms.SplitSourceRef(),
                              Transforms.Scale(), Transforms.Shear(), Transforms.Mirror(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetDeterministic(),
                              Transforms.FixedResampler(2048),
                              Transforms.SplitSourceRef(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.ShufflePoints()]
        elif noise_type == "jitter":
            # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
            if split == "train":
                transforms = [Transforms.SplitSourceRef(),
                              Transforms.Scale(), Transforms.Shear(), Transforms.Mirror(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.Resampler(2048),
                              Transforms.RandomJitter(),
                              Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetDeterministic(),
                              Transforms.SplitSourceRef(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.Resampler(2048),
                              Transforms.RandomJitter(),
                              Transforms.ShufflePoints()]
        else:
            raise ValueError(f"Noise type {noise_type} not supported for CustomData.")

        return torchvision.transforms.Compose(transforms)

    def get_samples(self, dataset_path, split):
        if split == 'train':
            path = os.path.join(dataset_path, 'train_data')
        elif split == 'val':
            path = os.path.join(dataset_path, 'val_data')
        else:
            path = os.path.join(dataset_path, 'test_data')
            
        all_data = []
        all_labels = []
        for item in glob.glob(path + '/*.obj'):
            mesh = o3d.io.read_triangle_mesh(item)
            pcd = mesh.sample_points_uniformly(number_of_points=2048)
    
            xyz = np.array(pcd.points)
            data = xyz.astype(np.float32)
            labels = 0

            all_data.append(data)
            all_labels.append(labels)

        return np.array(all_data), np.array(all_labels)
    


# In[29]:


if __name__ == '__main__':
    dataset = CustomDataset(split = 'train', noise_type='clean')
    print(len(dataset))
    print(dataset[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




