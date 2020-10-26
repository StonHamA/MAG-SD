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
    FilterDataset, relabel_dataset, XRayCenterCrop, XRayResizer, histeq,ZscoreNormalize, triDim, BalanceDataset, \
    COVID19_Localize, BalanceDataset30, NIH_ROI_Dataset


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
        if self.which_dataset == 'COVID_plus_NIH_localize':
            dataset, pneumonia_localize_dataset = eval('self.init_' + self.which_dataset + '()')
        else:
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

        if self.which_dataset == 'COVID_plus_NIH_localize':
            return train_set, val_set, dataset, pneumonia_localize_dataset
        else:
            return train_set, val_set, dataset

    def count_instance_num(self, dataset):
        ids = [str(i) for i in dataset.indices]
        labels = pd.DataFrame(dataset.dataset.labels)
        labels = labels.iloc[ids].values
        instance_nums = sum(labels)
        return  instance_nums

    def init_NIH_localize(self):
        dataset = NIH_ROI_Dataset(transform=self.transform,
                                      data_aug=self.augmentation,
                                      imgpath=os.path.join(self.dataset_path, 'NIH/images-224'),
                                      csvpath=os.path.join(self.dataset_path, 'NIH/Data_Entry_2017.csv'),
                                      seed=int(time.strftime('%H%M%S', time.localtime())),
                                      data_out_labels=["Nodule", "Mass", "Pneumonia", ],
                                      resize=self.image_size)
        dataset = BalanceDataset(dataset, least_label=None)
        return dataset

    def init_COVID_plus_NIH_localize(self):
        dataset_COVID = COVID19_Lung_Seg_Dataset(transform=self.transform,
                                           data_aug=self.augmentation,
                                           imgpath=os.path.join(self.dataset_path, 'COVID-19_with_lung_seg/images_with15'),
                                           metapath=os.path.join(self.dataset_path, 'COVID-19_with_lung_seg/labels_csv.csv'),
                                           maskpath=os.path.join(self.dataset_path, 'COVID-19_with_lung_seg/mask_images'),
                                           seg_flag=self.seg_flag,
                                           seed=int(time.strftime('%H%M%S', time.localtime())),
                                           data_out_labels=["Viral Pneumonia", "COVID-19", 'Healthy'] )
        dataset_NIH = NIH_ROI_Dataset(transform=self.transform,
                                  data_aug=self.augmentation,
                                  imgpath=os.path.join(self.dataset_path, 'NIH_localize/image'),
                                  csvpath=os.path.join(self.dataset_path, 'NIH_localize/BBox_List_2017.csv'),
                                  seed=int(time.strftime('%H%M%S', time.localtime())),
                                  data_out_labels=['Pneumonia'],
                                      resize=self.image_size)
        relabel_dataset(pathologies=["Viral Pneumonia","Pneumonia","COVID-19",'Healthy'], dataset=dataset_COVID)
        relabel_dataset(pathologies=["Viral Pneumonia","Pneumonia","COVID-19", 'Healthy'], dataset=dataset_NIH)
        dataset = Merge_Dataset((dataset_COVID, dataset_NIH), seed=1)

        label_replace = []
        for label in dataset.labels:
            if (label == np.array([1,0,0,0])).all():
                label_replace.append([0,1,0,0])
            else:
                label_replace.append(list(label))
        dataset.labels  = np.array(label_replace)
        relabel_dataset(pathologies=["Pneumonia","COVID-19",'Healthy'], dataset=dataset)
        relabel_dataset(pathologies=["Pneumonia","COVID-19",'Healthy'], dataset=dataset_NIH)
        dataset = BalanceDataset(dataset, least_label=None)
        return dataset, dataset_NIH



    def init_NIH(self):
        dataset = NIH_Dataset(
                 imgpath = os.path.join(self.dataset_path, "NIH", "images-224"),
                 csvpath=os.path.join(self.dataset_path, "NIH", "Data_Entry_2017.csv"),
                 transform=self.transform,
                 data_aug=self.augmentation,
                 nrows=None,
                 seed=int(time.strftime('%H%M%S', time.localtime())),
                 pure_labels=True,
                 unique_patients=True)
        dataset_roi = NIH_ROI_Dataset(transform=self.transform,
                                  data_aug=self.augmentation,
                                  imgpath=os.path.join('/home/jingxiongli/datasets/', 'NIH_localize/image'),
                                  csvpath=os.path.join('/home/jingxiongli/datasets', 'NIH_localize/BBox_List_2017.csv'),
                                  seed=int(time.strftime('%H%M%S', time.localtime())),
                                  data_out_labels=['Pneumonia'],
                                      resize=self.image_size)

        dataset = FilterDataset(dataset, labels=["Nodule", "Mass", "Pneumonia"])
        relabel_dataset(pathologies=["Nodule", "Mass", "Pneumonia"], dataset=dataset)
        relabel_dataset(pathologies=["Nodule", "Mass", "Pneumonia"], dataset=dataset_roi)
        dataset = Merge_Dataset((dataset, dataset_roi), seed=1)
        dataset = BalanceDataset(dataset, None)
        return dataset

    def init_COVID_localize_small(self):
        dataset = COVID19_Lung_Seg_Dataset(transform=self.transform,
                                           data_aug=self.augmentation,
                                           imgpath=os.path.join(self.dataset_path, 'images_with15'),
                                           metapath=os.path.join(self.dataset_path, 'labels_csv.csv'),
                                           maskpath=os.path.join(self.dataset_path, 'mask_images'),
                                           seg_flag=self.seg_flag,
                                           seed=int(time.strftime('%H%M%S', time.localtime())))
        dataset = BalanceDataset30(dataset, 'COVID-19')
        return dataset

    def init_COVID_localize(self):
        dataset = COVID19_Localize(transform=self.transform,
                                   data_aug=self.augmentation,
                                   imgpath=os.path.join(self.dataset_path, 'Imgs'),
                                   maskpath=os.path.join(self.dataset_path, 'Masks'),
                                   masktransform= torchvision.transforms.Compose([XRayResizer(224),
												                                  t.ToPILImage(),
                                                                                  t.CenterCrop(180),
                                                                                  t.Resize(224),
                                                                                  t.ToTensor()
												]),
                                   seed=int(time.strftime('%H%M%S', time.localtime())))
        return dataset

    def init_COVID_lungseg(self):
        dataset = COVID19_Lung_Seg_Dataset(transform=self.transform,
                                           data_aug=self.augmentation,
                                           imgpath=os.path.join(self.dataset_path, 'images_with15'),
                                           metapath=os.path.join(self.dataset_path, 'labels_csv.csv'),
                                           maskpath=os.path.join(self.dataset_path, 'mask_images'),
                                           seg_flag=self.seg_flag,
                                           seed=int(time.strftime('%H%M%S', time.localtime())),
                                           # data_out_labels=["Viral Pneumonia", "Bacterial Pneumonia"],
                                           data_out_labels=["Viral Pneumonia",  "Bacterial Pneumonia","COVID-19", 'Healthy']
                                           )
        dataset = BalanceDataset(dataset, least_label=None)
        return dataset

    def init_COVID_NIH(self):
        covid_dataset = COVID19_Dataset(
                 imgpath=os.path.join(self.dataset_path, "covid-chestxray-dataset", "images"),
                 csvpath=os.path.join(self.dataset_path, "covid-chestxray-dataset", "metadata.csv"),
                 views=["PA"],
                 transform=self.transform,
                 data_aug=self.augmentation,
                 nrows=None,
                 seed=time.strftime('%H%M%S', time.localtime()))
        NIH_dataset = NIH_Dataset(
                 imgpath = os.path.join(self.dataset_path, "NIH", "images-224"),
                 csvpath=os.path.join(self.dataset_path, "NIH", "Data_Entry_2017.csv"),
                 transform=self.transform,
                 data_aug=self.augmentation,
                 nrows=None,
                 seed=int(time.strftime('%H%M%S', time.localtime())),
                 pure_labels=True,
                 unique_patients=True)

        filtered_covid = FilterDataset(covid_dataset, labels=["COVID-19"])
        relabel_dataset(pathologies=["COVID-19", "x"], dataset=filtered_covid)
        relabel_dataset(pathologies=["COVID-19", "Pneumonia"], dataset=filtered_covid)
        filtered_NIH = FilterDataset(NIH_dataset, labels=['Pneumonia'])
        relabel_dataset(pathologies=["COVID-19", "Pneumonia"], dataset=filtered_NIH)
        dataset = Merge_Dataset((filtered_covid, filtered_NIH), seed=1)
        dataset = BalanceDataset(dataset, None)
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




if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms as t

    # Configurations
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str,
    #
    parser.add_argument('--dataset_path', type=str,
                        # default='/home/jingxiongli/PycharmProjects/lungDatasets')
                        # default = '/home/jingxiongli/datasets/')
                        # default = '/home/jingxiongli/datasets/COVID-19_with_lung_seg')
                        default = '/home/jingxiongli/PycharmProjects/lungDatasets')
    parser.add_argument('--which_dataset', type=str,
                        # default='COVID_NIH')
                        # default='COVID_plus_NIH_localize')
                        # default= 'COVID_lungseg')
                        default='NIH')

    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--image_size', type=int,
                        default=224)
    parser.add_argument('--seg_flag', type=bool, default=False)

    config = parser.parse_args()
    transform = torchvision.transforms.Compose([XRayResizer(224),
                                                t.ToPILImage(),
                                                ])
    # transform = None

    aug = torchvision.transforms.RandomApply([t.RandomRotation(180),
											  t.ColorJitter(brightness=0.5, contrast=0.7),
											  t.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33),
																  interpolation=2),
											  t.RandomHorizontalFlip(),
											  t.RandomVerticalFlip(),
                                              ], p=0.99)
    # ZscoreNormalize(), t.ToTensor()
    aug = torchvision.transforms.Compose([ZscoreNormalize(), t.ToTensor(),])
    lung_dataset = dataset_loader(config, transform=transform, augmentation=aug)

    # for i, item in enumerate(lung_dataset.all_set):
    #     # data = np.swapaxes(item[0], 0, -1)
    #     data = np.array(item[0])
    #     print(data.shape)
    #     label = item[1]
    #     print('label:', label)
    #     plt.imshow(np.squeeze(data), cmap='gray')
    #     plt.show()
    # print('0')

    x1, x2,  = lung_dataset.train_set_iter.next_one()
    print(x2)
    print(x1.size())
