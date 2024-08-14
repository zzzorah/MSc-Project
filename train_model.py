import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
# from torch.nn.utils import clip_grad_norm_
from torch.backends import cudnn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, \
    accuracy_score, recall_score, f1_score
from dataset import LIDCIDRIDataset
from tqdm import tqdm
from logs.logging_config import logger_epoch_result, logger_debug
from model.fusion_model import FusionModel

def train(train_data_loader, val_data_loader, epochs=1, use_gpu=False):
    model.train()
    results = []

    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0

        for inputs, features, targets in tqdm(train_data_loader):
            if use_gpu:
                inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
            targets -= 1

            optimizer.zero_grad()

            outputs = model(inputs, features)
            loss = loss_func(outputs[1], targets) # phi
            if torch.isnan(loss):
                logger_debug.debug(f'outputs={outputs}, targets={targets}, loss={loss}')
                raise ValueError(f'NaN {loss}')

            loss.backward()
            # max_norm = 1.0
            # clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            train_loss += loss
            _, predicted = torch.max(outputs[1], 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            torch.cuda.empty_cache()
        val_accuracy, val_loss, val_indicator = validate(val_data_loader, use_gpu=use_gpu)
        scheduler.step(val_loss)

        logger_epoch_result.info(f'Epoch: {epoch + 1:3}, Training Loss: {train_loss / total}, Training Accuracy: {correct / total}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, RMSE: {val_indicator[0]}, lr: {scheduler.get_last_lr()}')
        results.append([(train_loss / total).item(), correct / total, val_loss.item(), val_accuracy, *val_indicator])
        
    return results

def validate(data_loader, use_gpu: False):
    model.eval()
    with torch.no_grad():
        labels = []
        predictions = []
        loss = 0
        correct = 0
        total = 0
        for inputs, features, targets in data_loader:
            if use_gpu:
                inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
            targets -= 1

            inputs, features, targets = Variable(inputs), Variable(features), Variable(targets)

            outputs = model(inputs, features)
            loss =+ loss_func(outputs[1], targets)

            labels += targets.tolist()
            _, predicted = torch.max(outputs[1], 1)
            predictions += predicted.tolist()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        rmse = root_mean_squared_error(labels, predictions)
        # mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, average='macro')
        f1 = f1_score(labels, predictions, average='macro')

        torch.cuda.empty_cache()
        return correct/total, loss/total, [rmse, mae, r2, accuracy, recall, f1]

# dataset = LIDCIDRIDataset('dataset')
sequence = np.load(f'dataset/balanced_seg_ids.npy')
kfold = KFold(n_splits=10, shuffle=True)
epochs, lr, betas, batch_size = 400, 0.0005, (0.5, 0.999), 8

for fold, (train_idx, val_idx) in enumerate(kfold.split(sequence)):
    torch.cuda.empty_cache()
    train_dataset = Subset(LIDCIDRIDataset('dataset'), train_idx)
    val_dataset = Subset(LIDCIDRIDataset('dataset', is_train=False), val_idx)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FusionModel()
    device_ids = range(torch.cuda.device_count())
    model = nn.DataParallel(model, device_ids=device_ids)
    print('gpu use' + str(device_ids))
    cudnn.benchmark = False  # True

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5)

    result = train(train_data_loader, val_data_loader, epochs=epochs, use_gpu=bool(len(device_ids)))
    torch.save(result, f'results/lr({lr})-betas{betas}-epochs({epochs})-batch({batch_size})-kfold({fold}<{kfold}).pt')
    torch.save({'model': model,
            'epochs': epochs,
            'lr': lr,
            'batch_size': 8,
            'loss_func': {
                'method': 'cross entropy loss'
            },
            'optimizer':{
                'method': 'Adam',
                'lr': lr,
                'betas': betas
            },
            'adam_betas': betas,
            'scheduler':{
                'type': 'ReduceLROnPlateau',
                'mode': 'min',
                'factor': 0.95,
                'patience': 5
            }}, f'./models/kfold({fold}-10).pt')
 