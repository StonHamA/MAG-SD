import os
import time
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision
import pandas as pd
import torchvision.transforms as t
from dataset.covid19 import NIH_Dataset, COVID19_Dataset, Merge_Dataset, COVID19_Lung_Seg_Dataset, \
    FilterDataset, relabel_dataset, XRayCenterCrop, XRayResizer, histeq,ZscoreNormalize, triDim, BalanceDataset

class dataset_loader:
    def __init__(self, config, transform, augmentation):
        #  dataset configuration
        self.image_size = config.image_size
        self.dataset_path = config.dataset_path
        self.which_dataset = config.which_dataset
        self.seg_flag = config.seg_flag
        # transforms and augmentation
        self.transform = transform
        self.augmentation  = augmentation
        # init loaders
        if self.which_dataset == 'COVID_plus_NIH_localize':
            self.train_set, self.val_set, self.all_set, self.pneumonia_localize_dataset = self.init_sets()
        else:
            self.train_set, self.val_set, self.all_set = self.init_sets()

        self.train_set_loader = data.DataLoader(self.train_set, config.batch_size, shuffle=True, num_workers=2, drop_last=True)
        self.val_set_loader = data.DataLoader(self.val_set, config.batch_size, shuffle=True, num_workers=2, drop_last=True)
        self.all_set_loader = data.DataLoader(self.all_set, config.batch_size, shuffle=True, num_workers=2, drop_last=True)

        # init iters
        self.train_set_iter = IterLoader(self.train_set_loader)
        self.val_set_iter = IterLoader(self.val_set_loader)
        self.all_set_iter = IterLoader(self.all_set_loader)



    def init_sets(self):

        dataset = eval('self.init_'+self.which_dataset+'()')
        # spilt dataset
        train_set, val_set = data.random_split(dataset, [int(len(dataset)*4/5), len(dataset)-int(len(dataset)*4/5)],
                                               generator=torch.Generator().manual_seed(int(time.strftime('%H%M%S', time.localtime()))))
        pathologies = train_set.dataset.pathologies
        count_train = self.count_instance_num(train_set)
        count_val = self.count_instance_num(val_set)
        print('all_train', len(dataset), 'class:', pathologies, '  num:', count_train+count_val )
        print('train:', len(train_set), 'class:', pathologies, '  num:', count_train )
        print('validation:', len(val_set), 'class:', pathologies, '  num:', count_val )
        return train_set, val_set, dataset

    def count_instance_num(self, dataset):
        ids = [str(i) for i in dataset.indices]
        labels = pd.DataFrame(dataset.dataset.labels)
        labels = labels.iloc[ids].values
        instance_nums = sum(labels)
        return  instance_nums


    def init_COVID_lungseg(self):
        dataset = COVID19_Lung_Seg_Dataset(transform=self.transform,
                                           data_aug=self.augmentation,
                                           imgpath=os.path.join(self.dataset_path, 'images'),
                                           metapath=os.path.join(self.dataset_path, 'labels_csv.csv'),
                                           seed=int(time.strftime('%H%M%S', time.localtime())),
                                           data_out_labels=["Viral Pneumonia",  "Bacterial Pneumonia","COVID-19", 'Healthy']
                                           )
        dataset = BalanceDataset(dataset, least_label=None)
        return dataset

class IterLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)




