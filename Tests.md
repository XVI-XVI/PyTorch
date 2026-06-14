*тут я буду че то решать, скидывать решение и проводить работу над ошибками*
```Python
import time  
import matplotlib.pyplot as plt  
import matplotlib_inline.backend_inline  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.utils.data as data  
from matplotlib.colors import to_rgba  
from torch import Tensor  
from tqdm import tqdm  # Progress bar  
# matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export  
import torch  
  
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
  
  
class Net(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.linear1 = nn.Linear(2, hidden_size)  
        self.act_fn1 = nn.Sigmoid()  
        self.linear2 = nn.Linear(hidden_size, 1)  
  
    def forward(self, x):  
        x = self.linear1(x)  
        x = self.act_fn1(x)  
        x = self.linear2(x)  
        return x  
  
  
model = Net(40)  
  
# print(model)  
  
class XORdataset(data.Dataset):  
    def __init__(self, size, std=0.1):  
        super().__init__()  
  
        self.size = size  
        self.std = std  
        self.generate_XOR()  
  
    def generate_XOR(self):  
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)  
        label = (data.sum(dim=1) == 1).to(torch.long)  
        # label = torch.zeros(self.size, dtype=torch.long)  
        # indx = 0        # for i in data:        #     if i.sum() == 1:        #         label[indx] =1        #     else:        #         label[indx] =0        #     indx +=1        #     print(i)        #     print(label)        data += self.std * torch.randn(data.size(), dtype=torch.float32)  
  
        self.label = label  
        self.data = data  
  
    def __len__(self):  
        return self.size  
  
    def __getitem__(self, idx):  
        return self.data[idx], self.label[idx]  
  
  
train_dataset = XORdataset(100)  
# for i in range(len(train_dataset)):  
#     print(train_dataset[i])  
train_dataloader = data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)  
# for x, y in train_dataloader:  
#     print(x)  
#     print(y)  
  
loss_fn = nn.BCEWithLogitsLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
model.to(device)  
print("без обучения")  
  
with torch.no_grad():  
    x = int(input("№1: "))  
    x1 = int(input("№2: "))  
    inp = torch.tensor([x, x1], dtype=torch.float32, device=device).unsqueeze(0)  
  
    logits = model(inp)  
    prob = torch.sigmoid(logits)  
    pred = (prob > 0.5).int()  
  
    print(pred.item())  
  
# print(model(torch.tensor([x, x1], dtype=torch.float32)).unsqueeze(0))  
# inp = torch.tensor([x, x1], dtype=torch.float32).unsqueeze(0)  
# model(inp)  
def train_model(model, optimizer, data_loader, num_epochs=100):  
    model.train()  
    for epoch in tqdm(range(num_epochs)):  
        for x, y in data_loader:  
            x = x.to(device)  
            y = y.to(device)  
  
            preds = model(x)  
            preds = preds.squeeze(dim=1)  
  
            # preds = torch.sigmoid(preds)  
  
            loss = loss_fn(preds, y.to(torch.float))  
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
  
train_model(model, optimizer, train_dataloader)  
model.eval()  
while True:  
    with torch.no_grad():  
        x = int(input("№1: "))  
        x1 = int(input("№2: "))  
        inp = torch.tensor([x, x1], dtype=torch.float32, device=device).unsqueeze(0)  
  
        logits = model(inp)  
        prob = torch.sigmoid(logits)  
        pred = (prob > 0.5).int()  
  
        print(pred.item())
```