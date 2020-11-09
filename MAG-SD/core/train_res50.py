import torch

def train_res50_a_iter(config, base, loader, current_step):
    ### load data
    img_input, pid = loader.next_one()
    img_input, pid = img_input.float().to(base.device), pid.long().to(base.device)
    ### forward1
    logit_raw,_ = base.encoder(img_input)
    # print(logit_raw.shape)
    ### loss###
    acc_raw, loss = base.compute_classification_loss(logit_raw, pid)

    base.optimizer.zero_grad()
    loss.backward()
    base.optimizer.step()
    return ['acc_raw', 'loss_raw',], \
	       torch.Tensor([acc_raw[0], loss.data,])

def test_res50_a_iter(config, base, loader, current_step):
    with torch.no_grad():
        ### load data
        img_input, pid = loader.next_one()
        img_input, pid = img_input.float().to(base.device), pid.long().to(base.device)
        ### forward1
        logit_raw, map_out = base.encoder(img_input)
        ### loss
        acc_raw, loss_raw = base.compute_classification_loss(logit_raw, pid)
        test_name = ['test_acc_raw', 'test_loss_raw']
        test_value = torch.Tensor([acc_raw[0], loss_raw.data])
        pred_4sk = logit_raw.detach().cpu().numpy()
        pid_4sk = pid.detach().cpu().numpy()
        map_out = map_out.detach().cpu().numpy()

    return test_name, test_value, pred_4sk, pid_4sk , map_out