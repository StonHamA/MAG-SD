import torch
import numpy as np
import  os
import pandas
import torch.utils.data as data
import cv2
import torch.nn.functional as F
import torchvision.transforms as T
from core.utils import batch_augment
from dataset.covid19 import Dataset
from dataset.Dataloader import IterLoader
from dataset.covid19 import XRayCenterCrop, XRayResizer, CLAHE, ZscoreNormalize, triDim
import matplotlib.pyplot as plt
import pprint
from core.utils import generate_heatmap


def localize(config, base, loader, masks, start_epoch):
    # localize ids
    idx = [i + 3513 for i in range(15)]
    # loader.all_set.idxs.sort()
    imgs = torch.tensor(np.array([loader.all_set.dataset.__getitem__(id)[0].numpy() for id in idx ]))
    # for i, item in enumerate(loader.all_set):
    # 	img = item[0].unsqueeze(0)
    # 	mask = torch.tensor(item[2]).unsqueeze(0)
    # 	imgs = torch.cat((imgs, img), dim=0)
    # 	masks = torch.cat((masks, mask), dim=0)
    # print(imgs.shape)
    # print(masks.shape)

    with torch.no_grad():
        ### load data
        img_input = imgs.float().to(base.device)
        ### forward1
        logit_raw, feature_1, attention_map = base.encoder(img_input, False)
        mixup_images = batch_augment(img_input, attention_map[:, :1, :, :], mode='mixup', theta=(0.4, 0.6),
                                     padding_ratio=0.1)
        logit_mixup, _, _ = base.encoder(mixup_images, False)



        for i in range(15):
            print(logit_raw[i])
            CXR_img = img_input[i,0,:,:].detach().cpu().numpy()
            print('a,s',attention_map.shape())
            attention_map = generate_heatmap(attention_map)
            attention_map_1 = F.interpolate(attention_map[i:i+1,0:1,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
            # attention_map_1 = attention_map[i:i + 1, 0:1, :, :].squeeze().detach().cpu().numpy()
            mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (224, 224))

            attention_map_threshold = attention_map_1 > 0.75
            # attention_map_1 = attention_map[i:i + 1, 0:1, :, :].squeeze().detach().cpu().numpy()
            mask_IOU = mask[:, :] > 0
            intersection = attention_map_threshold & mask_IOU
            union = attention_map_threshold + mask_IOU

            intersection_n = np.sum(intersection.astype(np.int8))
            union_n = np.sum(union.astype(np.int8))
            IoU = intersection_n / union_n
            print(intersection_n, union_n, IoU)

            plt.subplot(2,3,1)
            plt.imshow(CXR_img, cmap='gray')
            plt.subplot(2,3,2)
            plt.imshow(attention_map_1, cmap='viridis', vmax=1.2, vmin=0) # vmax vmin用于指定颜色显示范围
            plt.colorbar(shrink=0.83)
            plt.subplot(2,3,3)
            plt.imshow(mask)
            plt.subplot(2, 3, 4)
            plt.imshow(attention_map_threshold)
            plt.subplot(2, 3, 5)
            plt.imshow(intersection)
            plt.subplot(2, 3, 6)
            plt.imshow(union)
            plt.show()
        test_name = 0
        test_value = 0
    return test_name, test_value

def localize_penumonia(config, base, loader, start_epoch):
    mask_path = './datasets/NIH_localize/mask'
    # transform the mask as it did to NIH dataset
    resize = T.Compose([XRayResizer(config.image_size),
                        T.ToPILImage(),
                        T.CenterCrop(170),
                        T.Resize(224),
                        T.ToTensor()
                        ])

    pneumonia_dataset_masked = MaskedNIH(loader.pneumonia_localize_dataset,mask_path, resize)
    pneumonia_dataset_masked_loader = data.DataLoader(pneumonia_dataset_masked, config.batch_size, shuffle=True, num_workers=2,
                                            drop_last=True)
    pneumonia_dataset_masked_iter = IterLoader(pneumonia_dataset_masked_loader)
    # x  = pneumonia_dataset_masked_iter.next_one()
    # for i, item in enumerate(pneumonia_dataset_masked):
    #     img = np.array(item[0])
    #     mask = np.array(item[1])
    #     print(img.shape)
    #     label = item[1]
    #     print('label:', label)
    #     plt.subplot(1,2,1)
    #     img = img * 0.7 + mask * 0.3
    #     plt.imshow(np.seeze(img), cmap='gray')
    #     plt.subplot(1,2,2)
    #     plt.imshow(np.squeeze(mask), cmap='gray')
    #     plt.show()

    for _ in range(30):
        with torch.no_grad():
            img_input, mask, pid = pneumonia_dataset_masked_iter.next_one()
            img_input, mask, pid = img_input.float().to(base.device), mask.long().to(base.device),pid.long().to(base.device)
            ### load data
            ### forward1
            logit_raw, feature_1, attention_map = base.encoder(img_input, False)

            mixup_images = batch_augment(img_input, attention_map[:, :1, :, :], mode='mixup', theta=(0.4, 0.6),
                                         padding_ratio=0.1)
            logit_mixup, _, _ = base.encoder(mixup_images, False)

            num_IoU_over30 = 0
            for i in range(16):
                x1 = 0.85
                x2 = 0.85
                # print(logit_raw[i])
                logit = logit_raw.detach().cpu().numpy()
                CXR_img = img_input[i,0,:,:].detach().cpu().numpy()
                attention_map_1 = F.interpolate(attention_map[i:i+1,0:1,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
                attention_map_threshold = attention_map_1 > x1
                attention_box = np.squeeze(mask2bbox(attention_map_threshold.astype(np.uint8))).astype(np.bool)
                # attention_map_1 = attention_map[i:i + 1, 0:1, :, :].squeeze().detach().cpu().numpy()
                mask_IOU = mask[i,0,:, :].detach().cpu().numpy() > 0
                intersection = attention_box & mask_IOU
                union = attention_box + mask_IOU
                AIoU = AttenIoU(attention_map_1, mask[i,0,:, :].detach().cpu().numpy())
                intersection_n = np.sum(intersection.astype(np.int8))
                union_n = np.sum(union.astype(np.int8))
                IoU = intersection_n / union_n
                m = mask[i, 0, :, :].detach().cpu().numpy()
                if IoU > 0.3:
                    num_IoU_over30 += 1
            # #
                plt.subplot(2,5,1)
                plt.imshow(CXR_img, cmap='gray')
                plt.imshow(m, alpha=0.3)
                plt.imshow(attention_box, alpha=0.4, cmap='RdBu')
                plt.axis('off')
                plt.subplot(2,5,2)
                plt.imshow(attention_map_1, cmap='viridis', vmax=1.1, vmin=x2)  # vmax vmin用于指定颜色显示范围
                plt.axis('off')
                # plt.colorbar(shrink=0.83)
                plt.subplot(2,5,3)
                plt.imshow(m)
                plt.axis('off')
                plt.subplot(2, 5, 4)
                plt.imshow(attention_box)
                plt.axis('off')
                plt.subplot(2, 5, 5)
                plt.imshow(intersection)
                plt.axis('off')
                plt.subplot(2, 5, 6)
                plt.imshow(union)
                plt.axis('off')
                plt.subplot(2,5,7)
                plt.imshow(CXR_img, cmap='gray')
                plt.imshow(m, alpha=0.3)
                plt.imshow(attention_map_threshold, alpha=0.4, cmap='RdBu')
                plt.axis('off')
                plt.subplot(2,5,8)
                plt.imshow(CXR_img, cmap='gray')
                plt.imshow(attention_box, alpha=0.4, cmap='RdBu')
                plt.axis('off')
                plt.subplot(2,5,9)
                plt.imshow(CXR_img, cmap='gray')
                plt.imshow(m, alpha=0.3)
                plt.axis('off')
                plt.subplot(2,5,10)
                plt.imshow(CXR_img, cmap='gray')
                plt.title('IOU:{}'.format(IoU))
                plt.axis('off')
                plt.show()

            num_acc_over30 = num_IoU_over30 / 16
            print(num_acc_over30)

    return pneumonia_dataset_masked_iter

def AttenIoU(Attention, Mask):
    attention_sigmoid = torch.sigmoid(torch.tensor(Attention)).cpu().numpy()
    x = attention_sigmoid * Mask
    AIoU = np.sum(attention_sigmoid * Mask)
    return AIoU





class MaskedNIH(Dataset):
    def __init__(self, dataset, dataset_path,Resize):
        super(MaskedNIH, self).__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.resize = Resize


    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset.__getitem__(idx)
        mask_id = self.dataset.csv['Image Index'].iloc[idx]
        mask_path = os.path.join(self.dataset_path, mask_id)
        # print(img_path)
        mask = cv2.imread(mask_path)

        # Check that images are 2D arrays
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        if len(mask.shape) < 2:
            print("error, dimension lower than 2 for image")

        if self.resize is not None:
            mask = self.resize(mask)
        return img, mask, label


def mask2bbox(maskImage):
    contours, hierarchy = cv2.findContours(maskImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 找到最大区域并填充
    # area = []
    # for j in range(len(contours)):
    #     area.append(cv2.contourArea(contours[j]))
    # max_idx = np.argmax(area)
    # max_area = cv2.contourArea(contours[max_idx])
    # for k in range(len(contours)):
    #     if k != max_idx:
    #         cv2.fillPoly(maskImage, [contours[k]], 0)

    image_out = np.zeros((224,224))
    bboxs = find_bbox(maskImage)
    for j in bboxs:
        cv2.rectangle(image_out,(j[0], j[1]),(j[0]+j[2], j[1]+j[3]), 255, thickness=-1)

    return image_out

def find_bbox(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]
