import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from torch.backends import cudnn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from dataset import LIDCIDRIDataset
from tqdm import tqdm
from logs.logging_config import logger_epoch_result, logger_predict
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
            inputs, features, targets = Variable(inputs), Variable(features), Variable(targets)

            outputs = model(inputs, features)
            loss = loss_func(outputs[1], targets) # phi

            loss.backward()
            max_norm = 1.0
            clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            train_loss += loss
            _, predicted = torch.max(outputs[1], 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            torch.cuda.empty_cache()
        val_accuracy, val_loss, val_rmse = validate(val_data_loader, use_gpu=use_gpu)
        scheduler.step(val_loss)

        logger_epoch_result.info(f'Epoch: {epoch + 1:3}, Training Loss: {train_loss / total}, Training Accuracy: {correct / total}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, RMSE: {val_rmse}, lr: {scheduler.get_last_lr()}')
        results.append([(train_loss / total).item(), correct / total, val_loss.item(), val_accuracy, val_rmse])
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
        torch.cuda.empty_cache()
        return correct/total, loss/total, rmse

# dataset = LIDCIDRIDataset('dataset')
sequence = np.load(f'dataset/balanced_seg_ids.npy')
kfold = KFold(n_splits=5, shuffle=True)
epochs, lr, betas, batch_size = 50, 0.0002, (0.6, 0.999), 8

for fold, (train_idx, val_idx) in enumerate(kfold.split(sequence)):
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=30, verbose=True)

    result = train(train_data_loader, val_data_loader, epochs=epochs, use_gpu=bool(len(device_ids)))
    torch.save(result, f'results/lr({lr})-betas{betas}-epochs({epochs})-batch({batch_size})-kfold({fold}<{kfold}).pt')
    # results.append(train(train_data_loader, val_data_loader, epochs=epochs, use_gpu=bool(len(device_ids))))

# torch.save(results, f'results/lr({lr})-betas{betas}-epochs({epochs})-batch({batch_size})-kfold({kfold}).pt')