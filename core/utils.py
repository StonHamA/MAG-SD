import numpy as np
import time
import torch
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, \
    precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics._classification import multilabel_confusion_matrix
from sklearn.metrics._classification import _prf_divide
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import sklearn
class Metric(object):
    pass


def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Successfully make dirs: {}'.format(dir))
    else:
        print('Existed dirs: {}'.format(dir))

def analyze_names_and_meter(loss_names, loss_meter):

    result = ''
    for i in range(len(loss_names)):

        loss_name = loss_names[i]
        loss_value = loss_meter[i]

        result += str(loss_name)
        result += ': '
        result += str(loss_value)
        result += ';  '

    return result

def analyze_meter_4_csv(loss_names, loss_meter):

    result = []
    for i in range(len(loss_names)):

        loss_value = round(loss_meter[i],3)
        result.append(loss_value)

    return result
## logger
class Logger:

    def __init__(self, logger_path):
        self.logger_path = logger_path

    def __call__(self, input, newline=True):
        input = str(input)
        if newline:
            input += '\n'

        with open(self.logger_path, 'a') as f:
            f.write(input)
            f.close()

        print(input)

# Meters
class AverageMeter:

    def __init__(self, neglect_value=None):
        self.reset()
        self.neglect_value = neglect_value

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n):
        if self.neglect_value is None or self.neglect_value not in val:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def get_val(self):
        return self.avg

    def get_val_numpy(self):
        return self.avg.data.cpu().numpy()

class CatMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        # print(val.shape)
        if self.val is None:
            self.val = val
        else:
            self.val = np.concatenate([self.val, val], axis=0)
    def get_val(self):
        return self.val

    def get_val_numpy(self):
        return self.val.data.cpu().numpy()

def batch_augment(images, attention_map, mode='mixup', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'mixup':
        auged_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            mixup_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c
            nonzero_indices = torch.nonzero(mixup_mask[0, 0, :,:], as_tuple =False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            upsampled_patch = F.interpolate(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW),  mode='bilinear')
            auged_image = images[batch_index:batch_index + 1,:,:,:]*0.6 + upsampled_patch*0.4
            # import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            # plt.imshow(upsampled_patch[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.subplot(2,1,2)
            # plt.imshow(auged_image[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.show()


            auged_images.append(auged_image)
        auged_images = torch.cat(auged_images, dim=0)
        return auged_images

    elif mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, :,:], as_tuple =False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            crop_images.append(F.interpolate(images[batch_index:batch_index + 1, :,
                                             height_min:height_max, width_min:width_max],
                                             size=(imgH, imgW),
                                             mode='bilinear'))

        crop_images = torch.cat(crop_images, dim=0)
        return crop_images


    elif mode == 'dim':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()
            # drop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d
            # drop_mask = drop_masks.float() * np.ones_like(drop_mask.numpy()) + 0.001
            # drop_masks.append(drop_mask)
            drop_masks.append(F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * (drop_masks.float() * torch.ones_like(drop_masks) + 0.001)
        return drop_images

    elif mode == 'patch':
        multi_image = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()
            crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear') >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, :,:], as_tuple =False)
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
            patch = images.clone()[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max]
            auged_image = images.clone()[batch_index:batch_index + 1, :, ...]
            H_patch = random.randint(0, imgH-(height_max-height_min))
            W_patch = random.randint(0, imgW-(width_max-width_min))
            auged_image[:, :,H_patch:H_patch+(height_max-height_min), W_patch:W_patch+(width_max-width_min)] = patch
            multi_image.append(auged_image)
            # import matplotlib.pyplot as plt
            # plt.subplot(2,1,1)
            # plt.imshow(patch[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.subplot(2,1,2)
            # plt.imshow(auged_image[0][0].squeeze().cpu().numpy(), cmap='gray')
            # plt.show()
        multi_images = torch.cat(multi_image, dim=0)
        return multi_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


def metrics_plots(score, pred, pid, current_step, config):
    # print(pred.shape)
    # print(pid)
    # print(pid.shape)
    pred = np.argmax(pred, axis=-1)
    pid = np.argmax(pid, axis=-1)
    confusion = confusion_matrix(pid, pred)
    acc = accuracy_score(pid,pred)
    precision = precision_score(pid,pred,average='macro')
    recall = recall_score(pid,pred,average='macro') #recall
    f1 = f1_score(pid,pred,average='macro')
    Specificity = SpecificityCalc(pid,pred)
    FPR = FPRCalc(pid,pred)

    # print(confusion)
    # TP = confusion[1, 1]
    # TN = confusion[0, 0]
    # FP = confusion[0, 1]
    # FN = confusion[1, 0]
    # Sensitivity = TP / float(TP + FN)
    # Specificity = TN / float(TN + FP)
    # Precision = precision_score(pid, pred)
    # Recall = recall_score(pid, pred)
    # F1 = f1_score(pid, pred)
    #
    # precision, recall, _ = precision_recall_curve(pid, score)
    # ap = average_precision_score(pid, score)
    # fpr, tpr, _ = roc_curve(pid, score)
    # plt.figure(0)
    # plt.plot(recall, precision, 'k--', color=(0.1, 0.9, 0.1), label='AP = {0:.2f}'.format(ap), lw=2)
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel("Recall Rate")
    # plt.ylabel("Precision Rate")
    # plt.legend(loc="lower right")
    # fig_name = 'PR_test' + str(current_step) + '.jpg'
    # plt.savefig(os.path.join(config.save_path, 'images', fig_name))
    # plt.clf()
    #
    # plt.figure(1)
    # Auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, 'k--', color=(0.1, 0.1, 0.9), label='Mean ROC (area = {0:.2f})'.format(Auc), lw=2)
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right")
    # fig_name = 'ROC_test' + str(current_step) + '.jpg'
    # plt.savefig(os.path.join(config.save_path, 'images', fig_name))
    # plt.clf()
    metric_bag = [acc, precision, recall, f1, Specificity, FPR]
    print(metric_bag)

    return confusion, metric_bag

def analyse_attention_vectors(attention_map, pids):
    attention_map = attention_map.squeeze().reshape(-1, 49) #7*7 attention map --> 49div attention vector
    pids = np.argmax(pids, -1) #1-hot to lables
    X_pca = TruncatedSVD(n_components=2).fit_transform(attention_map)
    RS = 20200101
    X_TSNE = TSNE(random_state=RS).fit_transform(attention_map)
    scatter(X_TSNE, pids)
    plt.show()

from matplotlib import offsetbox
def plot_embedding(X,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)#正则化
    print(X)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    # color = ['red', 'orange', 'yellow', 'green', 'cyan',
    #       'blue', 'purple', 'pink', 'magenta', 'brown']
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),


                 fontdict={'weight': 'bold', 'size': 12})
    # 打印彩色字体
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            # shown_images = np.r_[shown_images, [X[i]]]
            # imagebox = offsetbox.AnnotationBbox(
            #     offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
            #     X[i])
            # ax.add_artist(imagebox)  # 输出图上输出图片
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

import seaborn as sns
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("husl", 4))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def SpecificityCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    tn_sum = MCM[:, 0, 0]
    fp_sum = MCM[:, 0, 1]
    Condition_negative = tn_sum + fp_sum

    Specificity = tn_sum / Condition_negative
    Specificity = np.average(Specificity, weights=None)
    return Specificity

def FPRCalc(Predictions, Labels):
    MCM = multilabel_confusion_matrix(Labels, Predictions,
                                      sample_weight=None,
                                      labels=None, samplewise=None)
    tn_sum = MCM[:, 0, 0]
    fp_sum = MCM[:, 0, 1]
    Condition_negative = tn_sum + fp_sum

    FPR = fp_sum / Condition_negative
    FPR = np.average(FPR, weights=None)
    return FPR


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)