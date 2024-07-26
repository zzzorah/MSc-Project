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

def train(epoch):
    logger_debug.info(f'Start training. Epoch: {epoch}')
    model.train()
    # get_lr(epoch)
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_data_loader):
        # inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        # inputs = inputs.unsqueeze(1)
        logger_debug.debug(f'input shape = {inputs.shape}')

        outputs = model(inputs)
        loss = loss_func(outputs, targets)

        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'

    logger_epoch_result.info(f'Epoch: {epoch:3}, Training Accuracy: {correct.data.item() / float(total)}')
    logger_debug.info(f'Epoch: {epoch:3}, Training Accuracy: {correct.data.item() / float(total)}')


train_data_set = LIDCIDRIDataset('dataset')
train_data_loader = DataLoader(train_data_set, batch_size=1, shuffle=False)

model = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
device_ids = range(torch.cuda.device_count())
model = torch.nn.DataParallel(model, device_ids=device_ids)
print('gpu use' + str(device_ids))
cudnn.benchmark = False  # True

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

train(1)