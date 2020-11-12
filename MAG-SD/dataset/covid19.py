from os.path import join
from skimage.io import imread, imsave
from sklearn import preprocessing
import cv2
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import time
import numpy as np
import os, sys, os.path
import pandas as pd
import random
import collections
import pprint
import warnings

default_pathologies = ['Atelectasis',
                       'Consolidation',
                       'Infiltration',
                       'Pneumothorax',
                       'Edema',
                       'Emphysema',
                       'Fibrosis',
                       'Effusion',
                       'Pneumonia',
                       'Pleural_Thickening',
                       'Cardiomegaly',
                       'Nodule',
                       'Mass',
                       'Hernia',
                       'Lung Lesion',
                       'Fracture',
                       'Lung Opacity',
                       'Enlarged Cardiomediastinum'
                       ]

thispath = os.path.dirname(os.path.realpath(__file__))


def normalize(sample, maxval):
    return sample


def relabel_dataset(pathologies, dataset):
    """
    Reorder, remove, or add (nans) to a dataset's labels.
    Use this to align with the output of a network.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:, pathology_idx])
        else:
            print("{} doesn't exist. Adding 0 instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(0)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T

    dataset.labels = new_labels
    dataset.pathologies = pathologies


class Dataset():
    def __init__(self):
        pass

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")


class Merge_Dataset(Dataset):
    def __init__(self, datasets, seed=0, label_concat=False):
        super(Merge_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasets = datasets
        self.length = 0
        self.pathologies = datasets[0].pathologies
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0
        for i, dataset in enumerate(datasets):
            self.which_dataset = np.concatenate([self.which_dataset, np.zeros(len(dataset)) + i])
            self.length += len(dataset)
            self.offset = np.concatenate([self.offset, np.zeros(len(dataset)) + currentoffset])
            currentoffset += len(dataset)
            if dataset.pathologies != self.pathologies:
                raise Exception("incorrect pathology alignment")

        if hasattr(datasets[0], 'labels'):
            self.labels = np.concatenate([d.labels for d in datasets])
        else:
            print("WARN: not adding .labels")

        self.which_dataset = self.which_dataset.astype(int)

        if label_concat:
            new_labels = np.zeros([self.labels.shape[0], self.labels.shape[1] * len(datasets)]) * np.nan
            for i, shift in enumerate(self.which_dataset):
                size = self.labels.shape[1]
                new_labels[i, shift * size:shift * size + size] = self.labels[i]
            self.labels = new_labels

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.datasets[int(self.which_dataset[idx])][idx - int(self.offset[idx])]
        lab = self.labels[idx]
        source = self.which_dataset[idx]
        return item[0], lab


class FilterDataset(Dataset):
    def __init__(self, dataset, labels=None):
        super(FilterDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        #         self.idxs = np.where(np.nansum(dataset.labels, axis=1) > 0)[0]

        self.idxs = []
        if labels:
            for label in labels:
                print("filtering for ", label)
                self.idxs += list(np.where(dataset.labels[:, dataset.pathologies.index(label)] == 1)[0])
        self.labels = self.dataset.labels[self.idxs]

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

class BalanceDataset(Dataset):
    """
    convert all the classes of data to equal number
    dataset: target dataset
    label: the label want to be balanced (label has the least cases)
    """
    def __init__(self, dataset, least_label=None,):
        super(BalanceDataset, self).__init__()
        self.dataset = dataset
        self.least_label = least_label
        self.pathologies = dataset.pathologies
        if self.least_label is not None:
            self.idxs = self.limit_class_num_as_COVID()
            self.labels = self.dataset.labels[self.idxs]
        else:
            self.idxs = [num for num in range(len(dataset))]
            self.labels = dataset.labels
            self.x = 0

    def limit_class_num_as_COVID(self):
        labels = self.dataset.labels
        # labels = pd.DataFrame(dataset.labels)
        instance_nums = sum(labels)
        print(self.dataset.pathologies)
        covid_nums = int(instance_nums[self.dataset.pathologies.index(self.least_label)])
        # choose indices
        idx = []
        for i, name in enumerate(self.dataset.pathologies):
            label_opreate = np.zeros(len(self.dataset.pathologies))
            label_opreate[i] = 1
            # mask = list(labels_tup).index(tuple(label_opreate))
            index = [k for k in range(len(labels)) if (labels[k] == label_opreate).all()]
            idx += random.sample(index, covid_nums)
        return idx

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]




class NIH_Dataset(Dataset):
    """
    NIH ChestX-ray8 dataset
    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self, imgpath,
                 csvpath=os.path.join(thispath, "Data_Entry_2017.csv"),
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pure_labels=False,
                 unique_patients=True):

        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug

        # self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
        #                     "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
        #                     "Effusion", "Pneumonia", "Pleural_Thickening",
        #                     "Cardiomegaly", "Nodule", "Mass", "Hernia"]
        self.pathologies = [ "Pneumonia", "Nodule", "Mass"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Remove multi-finding images.
        if pure_labels:
            self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        # print(img_path)
        img = imread(img_path)
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        return img, self.labels[idx]


class NIH_14_Dataset(Dataset):
    """
    NIH ChestX-ray14 dataset

    """

    def __init__(self, imgpath,
                 csvpath=os.path.join(thispath, "Data_Entry_2017.csv"),
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pure_labels=False,
                 unique_patients=True):

        super(NIH_14_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug

        # self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
        #                     "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
        #                     "Effusion", "Pneumonia", "Pleural_Thickening",
        #                     "Cardiomegaly", "Nodule", "Mass", "Hernia",]
        self.pathologies = [ "Pneumonia""Nodule", "Mass",]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Remove multi-finding images.
        if pure_labels:
            self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        # print(img_path)
        img = imread(img_path)
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        return img, self.labels[idx]


class COVID19_Dataset(Dataset):
    """
    COVID-19 image data collection
    Dataset: https://github.com/ieee8023/covid-chestxray-dataset

    Paper: https://arxiv.org/abs/2003.11597
    """

    def __init__(self,
                 imgpath=os.path.join(thispath, "covid-chestxray-dataset", "images"),
                 csvpath=os.path.join(thispath, "covid-chestxray-dataset", "metadata.csv"),
                 views=['PA'],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pure_labels=False,
                 unique_patients=True):

        super(COVID19_Dataset, self).__init__()
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = views

        # defined here to make the code easier to read
        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella",
                      "Chlamydophila", "Legionella"]

        self.pathologies = ["Pneumonia", "Viral Pneumonia", "Bacterial Pneumonia", "Fungal Pneumonia",
                            "No Finding"] + pneumonias
        self.pathologies = sorted(self.pathologies)

        mapping = dict()
        mapping["Pneumonia"] = pneumonias
        mapping["Viral Pneumonia"] = ["COVID-19", "SARS", "MERS"]
        mapping["Bacterial Pneumonia"] = ["Streptococcus", "Klebsiella", "Chlamydophila", "Legionella"]
        mapping["Fungal Pneumonia"] = ["Pneumocystis"]

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Keep only the frontal views.
        # idx_pa = self.csv["view"].isin(["PA", "AP", "AP Supine"])
        idx_pa = self.csv["view"].isin(self.views)
        self.csv = self.csv[idx_pa]

        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["finding"].str.contains(syn)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={} views={}".format(len(self), self.views)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        if self.transform is not None:
            img = self.transform(img)
        if self.data_aug is not None:
            img = self.data_aug(img)

        return img, self.labels[idx]



class COVID19_Lung_Seg_Dataset(Dataset):

    def __init__(self,
                 imgpath='',
                 metapath='',
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 data_out_labels=None,
                 seed=0):

        super(COVID19_Lung_Seg_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = metapath


        # defined here to make the code easier to read
        if data_out_labels is None:
            # self.pathologies = ["Viral Pneumonia", "Bacterial Pneumonia",
            #                     'Fungal Pneumonia', "COVID-19", 'Healthy']
            self.pathologies = ["Viral Pneumonia", "Bacterial Pneumonia",
                                 "COVID-19", 'Healthy']
        else:
            self.pathologies = data_out_labels

        # Load data
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255

        # Keep only the cases with Pneumonia labels.
        idx_penumonia = self.csv["annotations"].isin(self.pathologies)
        self.csv = self.csv[idx_penumonia]

        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["annotations"].str.contains(pathology)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T.astype(np.float32)

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={} ".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        imgid = self.csv['id'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        if self.transform is not None:
            img = self.transform(img)
            # img = img.convert('L')
        if self.data_aug is not None:
            img = self.data_aug(img)
        labels = self.labels[idx]
        return  img, labels


class XRayResizer(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skimage.transform.resize(img, (1, self.size, self.size), mode='constant').astype(np.float32)
            return cv2.resize(img, (self.size, self.size))


class XRayCenterCrop(object):

    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size / 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


class ZscoreNormalize(object):
    def ZscoreNormalize(self, img):
        return preprocessing.scale(img)
    def __call__(self, img):
        return self.ZscoreNormalize(img)

class triDim(object):
    def __call__(self, input):
        return input.repeat(3,1,1)


class histeq(object):
    def histeq(self, im):
        """ Histogram equalization of a grayscale image. """
        im = np.array(im)
        imhist, bins = np.histogram(im.flatten(), 256)
        cdf = imhist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        im2 = np.interp(im.flatten(), bins[:-1], cdf)  # im2 is an array
        im2 = im2.reshape(im.shape)
        return im2

    def __call__(self, img):
        return self.histeq(img)


class CLAHE(object):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.clip_limit_tuple = to_tuple(clip_limit, 1)
        self.tile_grid_size = tuple(tile_grid_size)

    def __call__(self, img,):
        return clahe(img, self.clip_limit, self.tile_grid_size)

def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):

    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


if __name__ == '__main__':
    import torch.utils.data as data
    import matplotlib.pyplot as plt
    test_COVID_dataset = COVID19_Lung_Seg_Dataset(
                 imgpath = './datasets',
                 transform=None,
                 data_aug=None,)

    for i, item in enumerate(test_COVID_dataset):
        data = item[0]
        plt.subplot(2,1,1)
        plt.imshow(data, cmap='gray')
        mask = item[2]
        plt.subplot(2,1,2)
        plt.imshow(mask)
        plt.show()
        label = item[1]
        print(len(data.shape))
    print('0')