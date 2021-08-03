import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import time

class Net(nn.Module):
    def __init__(self, inden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inden, 60)
        self.fc2 = nn.Linear(60, 60)
        self.fc3 = nn.Linear(60, 1)
        self.dropout = nn.Dropout(0.2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = self.dropout(x)
        x = F.selu(self.fc2(x))
        x = self.dropout(x)
        x = F.selu(self.fc3(x))
        x = self.dropout(x)
        x = F.sigmoid(x)
        return x

def gettensor(device, df):
    if type(df) == pd.DataFrame:
        return torch.from_numpy(df.values).float().to(device)
    else:
        return torch.from_numpy(df).float().to(device)

def reweight(df):
    df.Event_Weight[df.label == 1] = df.Event_Weight[df.label == 1] * df.Event_Weight[df.label == 0].sum() / df.Event_Weight[df.label == 1].sum()

def main():
    device = torch.device('cpu')
    #device = torch.device('cuda:0')

    # load csv
    pd_bkg = pd.read_csv("bgLep.csv").rename(columns=lambda x: x.strip())
    pd_bkg.drop(columns=["Truth_Mass"], inplace=True)
    pd_bkg["Truth_Mass"] = np.random.choice([800, 1600], len(pd_bkg))

    pd_800 = pd.read_csv("800Lep.csv").rename(columns=lambda x: x.strip())
    pd_1600 = pd.read_csv("1600Lep.csv").rename(columns=lambda x: x.strip())
    signal_all = pd.concat([pd_800, pd_1600])

    pd_bkg['label'] = 0
    signal_all['label'] = 1

    # train test split
    train_bkg, test_bkg = train_test_split(pd_bkg, test_size=0.4, random_state=2)
    train_signal, test_signal = train_test_split(signal_all, test_size=0.4, random_state=2)
    val_bkg, test_bkg = train_test_split(test_bkg, test_size=0.5, random_state=2)
    val_signal, test_signal = train_test_split(test_signal, test_size=0.5, random_state=2)

    train_x = shuffle(pd.concat([train_bkg, train_signal], ignore_index=True))
    test_x = shuffle(pd.concat([test_bkg, test_signal], ignore_index=True))
    val_x = shuffle(pd.concat([val_bkg, val_signal], ignore_index=True))

    # reweight
    reweight(train_x)
    reweight(test_x)
    reweight(val_x)

    # get weight
    train_weight = train_x["Event_Weight"].to_numpy()
    train_y = train_x["label"].to_numpy()
    train_x.drop(columns=["Event_Weight", "label"], inplace=True)

    test_weight = test_x["Event_Weight"].to_numpy()
    test_y = test_x["label"].to_numpy()
    test_x.drop(columns=["Event_Weight", "label"], inplace=True)

    val_weight = val_x["Event_Weight"].to_numpy()
    val_y  = val_x["label"].to_numpy()
    val_x.drop(columns=["Event_Weight", "label"], inplace=True)

    # scale
    scaler = StandardScaler()
    # train_x_before = train_x
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    val_x = scaler.transform(val_x)

    # get tensor
    train_x = gettensor(device, train_x)
    train_y = gettensor(device, train_y)
    train_weight = gettensor(device, train_weight)

    val_x = gettensor(device, val_x)
    val_y = gettensor(device, val_y)
    val_weight = gettensor(device, val_weight)

    test_x = gettensor(device, test_x)
    test_y = gettensor(device, test_y)
    test_weight = gettensor(device, test_weight)

    # training
    batch_size = 100

    model = Net(train_x.shape[1])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    start = time.time()
    for epoch in range(1, 15):
        batch_count = 0
        permutation = torch.randperm(train_x.size()[0])
        model.train()
        for i in range(0, train_x.size()[0], batch_size):
            # get batch
            indices = permutation[i:i + batch_size]
            batch_x, batch_y, batch_weight = train_x[indices], train_y[indices], train_weight[indices]

            # define loss
            criterion = torch.nn.BCELoss(weight=batch_weight)

            # Backpropagation
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output[:, 0], batch_y)
            loss.backward()
            optimizer.step()

            batch_count +=1

        # calculate and print loss
        model.eval()
        output = model(val_x)
        criterion = torch.nn.BCELoss(weight=val_weight)
        val_loss = criterion(output[:, 0], val_y).item()

        output = model(train_x)
        criterion = torch.nn.BCELoss(weight=train_weight)
        train_loss = criterion(output[:, 0], train_y).item()
        print('Epoch: {} \tTraining Loss: {:.6f} \Validation Loss: {:.6f}'.format(
            epoch, train_loss, val_loss))

    end = time.time() - start
    print(end)
    output = model(test_x).detach().numpy()
    test_y = test_y.numpy()
    test_weight = test_weight.numpy()
    signal = output[test_y == 1]
    signal_weight = test_weight[test_y == 1]
    bkg = output[test_y == 0]
    bkg_weight = test_weight[test_y == 0]
    print(np.mean(bkg))
    print(np.mean(signal))

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
