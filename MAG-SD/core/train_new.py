import torch
import torch.nn.functional as F
from .utils import AverageMeter, time_now, batch_augment, CatMeter, metrics_plots, analyse_attention_vectors
from tqdm import tqdm
from .train_res50 import train_res50_a_iter, test_res50_a_iter


def train_a_ep(config, base, loader, current_step):
    base.set_train()
    train_meter = AverageMeter()
    # print(int(len(loader.train_set)/5))
    for i in tqdm(range(config.train_iter)):
        if config.Network == 'MAG-SD':
            train_titles, train_values = train_a_iter(config, base, loader.train_set_iter, current_step)
        else:
            train_titles, train_values = train_res50_a_iter(config, base, loader.train_set_iter, current_step)
        train_meter.update(train_values, 1)
    base.lr_decay(current_step)
    base.set_eval()
    return train_titles, train_meter.get_val_numpy()

def test_a_ep(config, base, loader, current_step):
    test_meter = AverageMeter()
    test_score_meter = CatMeter()
    test_pid_meter = CatMeter()
    test_pred_meter = CatMeter()
    attention_meter = CatMeter()

    base.set_eval()
    # print(len(loader.val_set))
    # range(int(len(loader.val_set)/2))
    for i in tqdm(range(config.test_iter)):
        if config.Network == 'MAG-SD':
            test_titles, test_values, test_pred, test_pid , test_attention= test_a_iter(config, base,loader.val_set_iter, current_step)
        else:
            test_titles, test_values, test_pred, test_pid , test_attention = test_res50_a_iter(config, base,loader.val_set_iter, current_step)
        test_meter.update(test_values, 1)
        test_pred_meter.update(test_pred)
        test_pid_meter.update(test_pid)
        # attention_meter.update(test_attention)


    score = test_score_meter.get_val()
    pred = test_pred_meter.get_val()
    pid = test_pid_meter.get_val()
    attention_vectors = attention_meter.get_val()
    confusion, metric_values = metrics_plots(score, pred, pid, current_step, config)
    # vector_analyzer = analyse_attention_vectors(attention_vectors, pid)

    # return test_titles, test_meter.get_val_numpy()
    return test_titles, test_meter.get_val_numpy(), confusion, metric_values




def train_a_iter(config, base, loader, current_step):
    ### load data
    img_input, pid = loader.next_one()
    img_input, pid = img_input.float().to(base.device), pid.long().to(base.device)
    # plt.imshow(img_input[0,0,:,:].detach().cpu().numpy())
    # plt.show()
    ### forward1
    logit_raw, feature_1,attention_map = base.encoder(img_input)

    ### batch augs
    # mixup
    mixup_images = batch_augment(img_input, attention_map[:, 0:1, :, :], mode='mixup', theta=(0.4, 0.6), padding_ratio=0.1)
    logit_mixup, _, _ = base.encoder(mixup_images)
    #
    # # dropping
    drop_images = batch_augment(img_input, attention_map[:, 1:2, :, :], mode='dim', theta=(0.2, 0.5))
    logit_dim, _, _ = base.encoder(drop_images)
    #
    # ## patching
    patch_images = batch_augment(img_input, attention_map[:, 2:3, :, :], mode='patch', theta=(0.4, 0.6), padding_ratio=0.1)
    logit_patch, _, _= base.encoder(patch_images)

    # attention plotting

    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     print(pid[i])
    #     print(logit_raw[i])
    #     CXR_img = img_input[i,0,:,:].detach().cpu().numpy()
    #     attention_map_1 = F.interpolate(attention_map[i:i+1,0:1,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
    #     attention_map_2 = F.interpolate(attention_map[i:i+1,1:2,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
    #     attention_map_3 = F.interpolate(attention_map[i:i+1,2:,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
    #     mixup_img = mixup_images[i,0,:,:].detach().cpu().numpy()
    #     drop_img = drop_images[i, 0, :, :].detach().cpu().numpy()
    #     patch_img = patch_images[i, 0, :, :].detach().cpu().numpy()
    #     plt.subplot(2,4,1)
    #     plt.imshow(CXR_img, cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(2,4,2)
    #     plt.imshow(attention_map_1)
    #     plt.axis('off')
    #     plt.subplot(2, 4, 3)
    #     plt.imshow(attention_map_2)
    #     plt.axis('off')
    #     plt.subplot(2, 4, 4)
    #     plt.imshow(attention_map_3)
    #     plt.axis('off')
    #     plt.subplot(2, 4, 5)
    #     plt.imshow(mixup_img, cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(2, 4, 6)
    #     plt.imshow(drop_img, cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(2, 4, 7)
    #     plt.imshow(patch_img, cmap='gray')
    #     plt.axis('off')
    #     plt.show()

    ### loss###
    acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
    acc_mixup, loss_mixup = base.compute_classification_loss(logit_mixup, pid)
    acc_dim, loss_dim = base.compute_classification_loss(logit_dim, pid)
    acc_patch, loss_patch = base.compute_classification_loss(logit_patch, pid)
    # L2 loss
    # loss = (loss_raw + loss_mixup + loss_dim + loss_patch)/4
    # logit_patch = 0
    # soft distance loss
    loss = base.gen_refine_loss(logit_raw, logit_mixup, logit_dim, logit_patch, pid)
    variance = loss - loss_raw
    base.optimizer.zero_grad()
    loss.backward()
    base.optimizer.step()

    return ['acc_raw', 'loss_raw', 'acc_mixup','loss_mixup','acc_dim','loss_dim','acc_patch','loss_patch','loss_v', 'variance'], \
	       torch.Tensor([acc_raw[0], loss_raw.data, acc_mixup[0], loss_mixup.data, acc_dim[0], loss_dim.data, acc_patch[0], loss_patch.data, loss, variance])
    # return ['acc_raw', 'loss_raw', 'acc_mixup','loss_mixup','acc_dim','loss_dim','loss_v', 'variance'], \
	#        torch.Tensor([acc_raw[0], loss_raw.data, acc_mixup[0], loss_dim.data, acc_dim[0], loss_mixup.data, loss, variance])
    # return ['acc_raw', 'loss_raw', 'acc_mixup','loss_mixup','loss_v', 'variance'], \
	#        torch.Tensor([acc_raw[0], loss_raw.data, acc_mixup[0], loss_mixup.data, loss, variance])
    # return ['acc_raw', 'loss_raw', ], \
    #        torch.Tensor(
    #            [acc_raw[0], loss_raw.data])




def test_a_iter(config, base, loader, current_step):
    with torch.no_grad():
        ### load data
        img_input, pid = loader.next_one()
        img_input, pid = img_input.float().to(base.device), pid.long().to(base.device)
        ### forward1
        logit_raw, feature_1, attention_map = base.encoder(img_input, False)

        mixup_images = batch_augment(img_input, attention_map[:, :1, :, :], mode='mixup', theta=(0.4, 0.6),
                                     padding_ratio=0.1)
        logit_mixup, _, _ = base.encoder(mixup_images, False)
        ### loss
        acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
        # print(acc_raw)
        acc_attention, loss_attention = base.compute_classification_loss(logit_mixup,  pid)
        acc = (acc_raw[0] + acc_attention[0])/2
        ### metrics
        # logit_mean = (logit_raw + logit_mixup)/2
        # score_4sk = []
        # print(pid)
        # print(logit_raw)
        # import matplotlib.pyplot as plt
        # for i in range(10):
        #     print(pid[i])
        #     print(logit_raw[i])
        #     CXR_img = img_input[i,0,:,:].detach().cpu().numpy()
        #     attention_map_1 = F.interpolate(attention_map[i:i+1,0:1,:,:], size=224, mode='bilinear').squeeze().detach().cpu().numpy()
        #     # attention_map_1 = attention_map[i:i + 1, 0:1, :, :].squeeze().detach().cpu().numpy()
        #     plt.subplot(2,2,1)
        #     plt.imshow(CXR_img, cmap='gray')
        #     plt.subplot(2,2,2)
        #     plt.imshow(attention_map_1, cmap='viridis', vmax=2,  vmin=1, )  # vmax vmin用于指定颜色显示范围
        #     plt.colorbar(shrink=0.83)
        #     plt.show()


        pred_4sk = logit_raw.detach().cpu().numpy()
        pid_4sk = pid.detach().cpu().numpy()
        attention_map_out = attention_map.detach().cpu().numpy()
        # for i, score_id in  enumerate(pid_4sk):
        #         score_4sk.append(logit_mean[i][1].detach().cpu().numpy())
        # score_4sk = np.array(score_4sk)
        test_name = ['test_acc_raw', 'test_loss_raw', 'test_acc_attention', 'test_loss_attention','acc']
        test_value = torch.Tensor([acc_raw[0], loss_raw.data, acc_attention[0], loss_attention.data, acc])
    return test_name, test_value, pred_4sk, pid_4sk, attention_map_out
