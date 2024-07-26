import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.backends import cudnn
from model.model import ConvRes
from dataset import LIDCIDRIDataset
from tqdm import tqdm
from logs.logging_config import logger_epoch_result, logger_debug
from model.ordinal_loss import OrdinalLoss

def train(epoch=1, use_gpu=False):
    logger_debug.info(f'Start training. Epoch: {epoch}')
    model.train()
    # get_lr(epoch)
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_data_loader):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        logger_debug.debug(f'outputs = {outputs}, targets = {targets}')
        logger_debug.debug(f'outputs type = {type(outputs)}')
        loss = loss_func(outputs, targets)

        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'

    logger_epoch_result.debug(f'Predict = {predicted}, Target = {targets}')
    logger_epoch_result.info(f'Epoch: {epoch:3}, Loss: {loss}, Training Accuracy: {correct.data.item() / float(total)}')
    logger_debug.info(f'Epoch: {epoch:3}, Loss: {loss}, Training Accuracy: {correct.data.item() / float(total)}')


train_data_set = LIDCIDRIDataset('dataset')
train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=False)

model = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
device_ids = range(torch.cuda.device_count())
model = torch.nn.DataParallel(model, device_ids=device_ids)
print('gpu use' + str(device_ids))
cudnn.benchmark = False  # True

loss_func = OrdinalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

train(epoch=1, use_gpu=bool(len(device_ids)))