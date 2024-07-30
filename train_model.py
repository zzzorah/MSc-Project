import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.backends import cudnn
from model.cnn import ConvRes
from dataset import LIDCIDRIDataset
from tqdm import tqdm
from logs.logging_config import logger_epoch_result, logger_predict
from model.fusion_model import FusionModel

def train(epochs=1, use_gpu=False):
    # logger_debug.info(f'Start training. Epochs: {epochs}')
    model.train()
    # get_lr(epoch)

    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0

        for inputs, features, targets in tqdm(train_data_loader):
            if use_gpu:
                inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
            targets -= 1

            optimizer.zero_grad()
            inputs, features, targets = Variable(inputs), Variable(features), Variable(targets)

            outputs = model(inputs, features)
            loss = loss_func(outputs[1], targets) # phi

            loss.backward()
            optimizer.step()

            train_loss += loss
            _, predicted = torch.max(outputs[1], 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        logger_predict.info(f'Epoch: {epoch + 1:3} - (Predictions, targets): {predicted}, {targets}')
        logger_epoch_result.info(f'Epoch: {epoch + 1:3}, Loss: {train_loss / total}, Training Accuracy: {correct / total}')


train_data_set = LIDCIDRIDataset('dataset')
train_data_loader = DataLoader(train_data_set, batch_size=8, shuffle=False)

model = FusionModel()
device_ids = range(torch.cuda.device_count())
model = nn.DataParallel(model, device_ids=device_ids)
print('gpu use' + str(device_ids))
cudnn.benchmark = False  # True

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.6, 0.999))

train(epochs=50, use_gpu=bool(len(device_ids)))