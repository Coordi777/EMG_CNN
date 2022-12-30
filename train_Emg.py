import random
import numpy as np
import torch
from pysiology import electromyography as emg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import CnnNet, EmgDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set the random seed
def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f'{seed} as random seed!')


# Transfer raw data from .mat to numpy
def mat2numpy(raw_x):
    train_list = []
    for k in range(len(raw_x)):
        train_list.append(raw_x[k])
    x_train = np.stack(train_list)
    x_train = x_train.transpose([0, 2, 1])
    return x_train


# Normalization or Standardization
def scaler(X, method='Minmax'):
    if method == 'Minmax':
        sc = MinMaxScaler(feature_range=(0, 1), copy=False)
    else:
        sc = StandardScaler(copy=False)
    l, h, w = X.shape
    X = X.reshape((l, -1))
    X = sc.fit_transform(X).reshape((-1, h, w))
    torch.from_numpy(X).to(torch.float32)
    return torch.from_numpy(X).to(torch.float32)


# Extract features
def feature_process(x_pre):
    print('Features are extracted...')
    data_list = []
    L1 = len(x_pre)
    L2 = len(x_pre[0])
    for m in tqdm(range(L1)):
        d = []
        for n in range(L2):
            data = x_pre[m][n]
            data_std = emg.getDASDV(data)  # standard_deviation
            data_acc = emg.getAAC(data)  # average_amplitude_change
            data_wl = emg.getWL(data)  # wave_form_length
            data_rms = emg.getRMS(data)  # root_mean_square
            data_mav = emg.getMAV(data)  # mean_absolute_value
            data_zc = emg.getZC(data, 0)  # zero_crossings
            data_max = data.max()  # maximum
            data_min = data.min()  # minimum
            data_wamp = emg.getWAMP(data, 50)  # Willison amplitude
            tmp = np.array(
                [data_std, data_rms, data_min, data_max, data_zc, data_acc, data_mav, data_wl, data_wamp])
            d.append(tmp)
        data_list.append(np.stack(d))
    fin = np.stack(data_list)
    return fin


# Sliding window processing
def win_process(x_raw, win_size=512, stride=64):
    x_numpy = mat2numpy(x_raw)
    win_list = []
    print(f'Start data augmentation with window size:{win_size} and stride:{stride}.')
    for k in range(len(x_numpy)):
        start = 0
        while True:
            tmp = x_numpy[k, :, start:start + win_size]
            win_list.append(tmp)
            start = start + stride
            if start > (2048 - win_size):
                break
    res = np.stack(win_list)
    fea = feature_process(res)
    return fea


# To train!
def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs=5):
    for epoch in range(epochs):
        for idx, (X_train, y_train) in enumerate(train_dataloader):
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_hat = model(X_train)
            loss = loss_fn(y_hat, y_train.long())
            if idx == 0 or (idx + 1) % 100 == 0:
                print(f'Epoch {epoch + 1} train loss: {loss.item()}')
                test(test_dataloader, model, loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


# Test
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    model.train()


def train_pattern1(train_x, train_y):
    set_seed()
    print('A new train on only motion set is beginning....')
    print('Start data process!')
    train_x = win_process(train_x, win_size=256, stride=32)
    x = scaler(train_x)
    train_y = train_y - 1
    y_repeat = len(x) / 30
    train_y = np.repeat(train_y, y_repeat, axis=0)
    y = torch.from_numpy(train_y).to(torch.float32).squeeze(-1)
    print('Data process is over.')
    print('Preparing training...')
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=3407)
    train_dataset = EmgDataset(X_train, y_train)
    test_dataset = EmgDataset(X_test, y_test)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=256)
    model = CnnNet(num_class=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    print("Start train!")
    print("Total epochs:", epochs)
    print('Train on', device)
    mdl = train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs)
    if device != 'cpu':
        mdl.to('cpu')
    return mdl


def predict_pattern1(mdl, val_x):
    print('Data process on Val.')
    print('Val on', device)
    val_x = mat2numpy(val_x)
    val_x = feature_process(val_x)
    val_x = scaler(val_x)
    X_val = val_x.to(device)
    mdl.eval()
    with torch.no_grad():
        mdl.to(device)
        pred = mdl(X_val)
        y_pred = pred.argmax(1)
        y_pred = y_pred.cpu().numpy() + 1
    return y_pred


def train_pattern2(train_x, train_x_rest, train_y):
    set_seed()
    print('A new train on motion&rest set is beginning....')
    print('Start data process!')
    x_all = np.concatenate((train_x, train_x_rest))
    x_all = win_process(x_all, win_size=1024, stride=64)
    l = len(x_all)
    x_m = x_all[:l // 2]
    x_r = x_all[l // 2:]
    x_m = scaler(x_m, method='Std')
    x_r = scaler(x_r, method='Std')
    x = np.concatenate((x_m, x_r))
    train_y = train_y - 1
    y_repeat = len(x_m) / 30
    train_y = np.repeat(train_y, y_repeat, axis=0)
    y_rest = np.repeat(10, len(x_m), axis=0)
    train_y = np.concatenate((train_y, y_rest))
    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(train_y).to(torch.float32).squeeze(-1)
    print('Data process is over.')
    print('Preparing training...')
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=3407)
    train_dataset = EmgDataset(X_train, y_train)
    test_dataset = EmgDataset(X_test, y_test)
    weights = [1 / 50 if i != 10 else 1 / 500 for i in y_train]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, sampler=sampler)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16)
    model = CnnNet(num_class=11).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 30
    print("Start train!")
    print("Total epochs:", epochs)
    print('Train on', device)
    mdl = train(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs)
    if device != 'cpu':
        mdl.to('cpu')
    return mdl


def predict_pattern2(mdl, val_x):
    print('Data process on Val.')
    print('Val on', device)
    val_x = mat2numpy(val_x)
    val_x = feature_process(val_x)
    val_x = scaler(val_x, method='Std')
    X_val = val_x.to(device)
    mdl.eval()
    with torch.no_grad():
        mdl.to(device)
        pred = mdl(X_val)
        y_pred = pred.argmax(1)
        y_pred = y_pred.cpu().numpy() + 1
    return y_pred
