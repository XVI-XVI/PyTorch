*тут я буду че то решать, скидывать решение и проводить работу над ошибками*

# XOR модель
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
# X + Y + Z Model
тупая нейронка (умножение), я ее много раз учил, менял ей данные, но пока что это лучший вариант
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
  
  
class SumNet(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.linear1 = nn.Linear(3, hidden_size)  
        self.fn = nn.ReLU()  
        self.linear2 = nn.Linear(hidden_size, hidden_size)  
        self.linear3 = nn.Linear(hidden_size, 1)  
  
    def forward(self, x):  
        x = self.linear1(x)  
        x = self.fn(x)  
        x = self.linear2(x)  
        x = self.fn(x)  
        x = self.linear3(x)  
        return x  
  
  
model = SumNet(64)  
  
model.to(device)  
# print("БЕЗ")  
# with torch.no_grad():  
#     x = float(input())  
#     y = float(input())  
#     z = float(input())  
#  
#     inp = torch.tensor([x, y, z], dtype=torch.float32, device=device).unsqueeze(0)  
#     print(model(inp).item())  
  
  
class SUMDataSet(data.Dataset):  
    def __init__(self, size):  
        self.size = size  
  
        self.generate_SUM()  
  
    def generate_SUM(self):  
        data = torch.randn(self.size,3,dtype=torch.float32) * 2  
  
        label = data.prod(1).to(torch.float)  
  
        self.data = data  
        self.label = label  
  
    def __len__(self):  
        return self.size  
  
    def __getitem__(self, indx):  
        return self.data[indx], self.label[indx]  
  
  
train_data = SUMDataSet(10000)  
# for i in range(len(train_data)):  
#     print(train_data[i])  
data_train_loader = data.DataLoader(train_data, batch_size=256, shuffle=True, drop_last=True)  
loss_fn = nn.MSELoss()  
optimazer = torch.optim.Adam(model.parameters(), lr=0.001)  
  
def train_model(model, optimizer, data_loader, num_epochs = 200):  
    model.train()  
    for epoch in tqdm(range(num_epochs)):  
        for data_inputs, data_label in data_loader:  
            data_inputs = data_inputs.to(device)  
            data_label = data_label.to(device)  
  
            preds = model(data_inputs).squeeze(1)  
  
            loss = loss_fn(preds, data_label)  
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
  
train_model(model, optimazer, data_train_loader)  
# for name, param in model.named_parameters():  
#     print(name, param)  
  
model.eval()  
while True:  
    with torch.no_grad():  
        x = float(input())  
        y = float(input())  
        z = float(input())  
  
        inp = torch.tensor([x, y, z], dtype=torch.float32, device=device).unsqueeze(0)  
        print(model(inp).item())
```
## Что то изменил:
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
count = 1  
  
  
class SumNet(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.linear1 = nn.Linear(3, hidden_size)  
        self.fn = nn.ReLU()  
        self.linear2 = nn.Linear(hidden_size, hidden_size)  
        self.linear3 = nn.Linear(hidden_size, 1)  
  
    def forward(self, x):  
        x = self.linear1(x)  
        x = self.fn(x)  
        x = self.linear2(x)  
        x = self.fn(x)  
        x = self.linear3(x)  
        return x  
  
  
model = SumNet(100)  
  
model.to(device)  
  
  
def test():  
    while True:  
        with torch.no_grad():  
            try:  
                x = input("№1: ")  
                if x.lower() == "s" or x.lower() == "ы":  
                    break  
                x = float(x)  
  
                y = input("№2: ")  
                if y.lower() == "s" or y.lower() == "ы":  
                    break  
                y = float(y)  
  
                z = input("№3: ")  
                if z.lower() == "s" or z.lower() == "ы":  
                    break  
                z = float(z)  
            except ValueError:  
                continue  
            inp = torch.tensor([x, y, z], dtype=torch.float32, device=device).unsqueeze(0)  
            print(model(inp).item())  
  
  
print("--- БЕЗ обучения ---")  
test()  
  
  
class SUMDataSet(data.Dataset):  
    def __init__(self, size):  
        self.size = size  
  
        self.generate_SUM()  
  
    def generate_SUM(self):  
        data = torch.randn(self.size, 3, dtype=torch.float32) * 2  
  
        label = data.prod(1).to(torch.float)  
  
        self.data = data  
        self.label = label  
  
    def __len__(self):  
        return self.size  
  
    def __getitem__(self, indx):  
        return self.data[indx], self.label[indx]  
  
  
train_data = SUMDataSet(20000)  
# for i in range(len(train_data)):  
#     print(train_data[i])  
data_train_loader = data.DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True)  
loss_fn = nn.MSELoss()  
optimazer = torch.optim.Adam(model.parameters(), lr=0.01)  
  
  
def train_model(model, optimizer, data_loader, num_epochs=100):  
    model.train()  
    print(f"    |Обучение №{count}|")  
    for epoch in tqdm(range(num_epochs)):  
        for data_inputs, data_label in data_loader:  
            data_inputs = data_inputs.to(device)  
            data_label = data_label.to(device)  
  
            preds = model(data_inputs).squeeze(1)  
  
            loss = loss_fn(preds, data_label)  
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
    model.eval()  
  
  
train_model(model, optimazer, data_train_loader)  
# for name, param in model.named_parameters():  
#     print(name, param)  
  
print("--- после обучения №1 ---")  
test()  
count += 1  
  
optimazer = torch.optim.Adam(model.parameters(), lr=0.001)  
train_model(model, optimazer, data_train_loader)  
print("--- после обучения №2 ---")  
test()  
count += 1  
  
  
optimazer = torch.optim.Adam(model.parameters(), lr=0.0001)  
data_train_loader = data.DataLoader(train_data, batch_size=1000, shuffle=True, drop_last=True)  
train_model(model, optimazer, data_train_loader, 50)  
print("--- после обучения №3 ---")  
test()
```

## версия с тестом
```Python
import time  
# import matplotlib.pyplot as plt  
# import matplotlib_inline.backend_inline  
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
count = 1  
  
  
class SumNet(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.linear1 = nn.Linear(3, hidden_size)  
        self.fn = nn.ReLU()  
        self.linear2 = nn.Linear(hidden_size, hidden_size)  
        self.linear3 = nn.Linear(hidden_size, 1)  
  
    def forward(self, x):  
        x = self.linear1(x)  
        x = self.fn(x)  
        x = self.linear2(x)  
        x = self.fn(x)  
        x = self.linear3(x)  
        return x  
  
  
model = SumNet(100)  
  
model.to(device)  
  
  
def test():  
    while True:  
        with torch.no_grad():  
            try:  
                x = input("№1: ")  
                if x.lower() == "s" or x.lower() == "ы":  
                    break  
                x = float(x)  
  
                y = input("№2: ")  
                if y.lower() == "s" or y.lower() == "ы":  
                    break  
                y = float(y)  
  
                z = input("№3: ")  
                if z.lower() == "s" or z.lower() == "ы":  
                    break  
                z = float(z)  
            except ValueError:  
                continue  
            inp = torch.tensor([x, y, z], dtype=torch.float32, device=device).unsqueeze(0)  
            print(model(inp).item())  
  
  
print("--- БЕЗ обучения ---")  
test()  
  
  
class SUMDataSet(data.Dataset):  
    def __init__(self, size):  
        self.size = size  
  
        self.generate_SUM()  
  
    def generate_SUM(self):  
        data = torch.randn(self.size, 3, dtype=torch.float32) * 2  
  
        label = data.prod(1).to(torch.float)  
  
        self.data = data  
        self.label = label  
  
    def __len__(self):  
        return self.size  
  
    def __getitem__(self, indx):  
        return self.data[indx], self.label[indx]  
  
  
train_data = SUMDataSet(20000)  
# for i in range(len(train_data)):  
#     print(train_data[i])  
data_train_loader = data.DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True)  
loss_fn = nn.MSELoss()  
optimazer = torch.optim.Adam(model.parameters(), lr=0.01)  
  
  
def train_model(model, optimizer, data_loader, num_epochs=100):  
    model.train()  
    print(f"    |Обучение №{count}|")  
    for epoch in tqdm(range(num_epochs)):  
        for data_inputs, data_label in data_loader:  
            data_inputs = data_inputs.to(device)  
            data_label = data_label.to(device)  
  
            preds = model(data_inputs).squeeze(1)  
  
            loss = loss_fn(preds, data_label)  
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
    model.eval()  
  
  
def test_model(model, data_loader):  
    model.eval()  
    total_accuracy, num_preds = 0.0, 0.0  
  
    with torch.no_grad():  
        for data_inputs, data_label in data_loader:  
            data_inputs, data_label = data_inputs.to(device), data_label.to(device)  
            preds = model(data_inputs).squeeze(1)  
  
            p = torch.abs(preds) + 1e-5  
            l = torch.abs(data_label) + 1e-5  
  
            accuracy_batch = torch.minimum(p, l) / torch.maximum(p, l)  
  
            total_accuracy += accuracy_batch.sum().item()  
            num_preds += data_label.shape[0]  
  
    mean_accuracy = total_accuracy / num_preds  
    print(f"\nAccuracy of the model: {100.0 * mean_accuracy:4.2f}%\n")  
  
data_test_loader = data.DataLoader(train_data, batch_size=512, shuffle=False, drop_last=True)  
test_model(model, data_test_loader)  
train_model(model, optimazer, data_train_loader)  
# for name, param in model.named_parameters():  
#     print(name, param)  
  
print("--- после обучения №1 ---")  
# test()  
count += 1  
test_model(model, data_test_loader)  
  
optimazer = torch.optim.Adam(model.parameters(), lr=0.001)  
train_model(model, optimazer, data_train_loader)  
print("--- после обучения №2 ---")  
# test()  
count += 1  
test_model(model, data_test_loader)  
  
optimazer = torch.optim.Adam(model.parameters(), lr=0.0001)  
data_train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)  
train_model(model, optimazer, data_train_loader, 50)  
print("--- после обучения №3 ---")  
# test()  
test_model(model, data_test_loader)
```

что бы уменьшить % ошибки, я сменил функцию активации c `ReLU` на `GELU`

```Python
....
class SumNet(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.linear1 = nn.Linear(3, hidden_size)  
        self.fn = nn.GELU()  
        self.linear2 = nn.Linear(hidden_size, hidden_size)  
        self.linear3 = nn.Linear(hidden_size, 1)
        
        
....
```
итог, процент с ~89%, вырос до ~94%!

## Итоговый код

```Python
import time  
# import matplotlib.pyplot as plt  
# import matplotlib_inline.backend_inline  
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
count = 1  
  
  
class SumNet(nn.Module):  
    def __init__(self, hidden_size):  
        super().__init__()  
        self.linear1 = nn.Linear(3, hidden_size)  
        self.fn = nn.GELU()  
        self.linear2 = nn.Linear(hidden_size, hidden_size)  
        self.linear3 = nn.Linear(hidden_size, 1)  
  
    def forward(self, x):  
        x = self.linear1(x)  
        x = self.fn(x)  
        x = self.linear2(x)  
        x = self.fn(x)  
        x = self.linear3(x)  
        return x  
  
  
model = SumNet(100)  
  
model.to(device)  
  
  
def test():  
    while True:  
        with torch.no_grad():  
            try:  
                x = input("№1: ")  
                if x.lower() == "s" or x.lower() == "ы":  
                    break  
                x = float(x)  
  
                y = input("№2: ")  
                if y.lower() == "s" or y.lower() == "ы":  
                    break  
                y = float(y)  
  
                z = input("№3: ")  
                if z.lower() == "s" or z.lower() == "ы":  
                    break  
                z = float(z)  
            except ValueError:  
                continue  
            inp = torch.tensor([x, y, z], dtype=torch.float32, device=device).unsqueeze(0)  
            print(model(inp).item())  
  
  
print("--- БЕЗ обучения ---")  
test()  
  
  
class SUMDataSet(data.Dataset):  
    def __init__(self, size):  
        self.size = size  
  
        self.generate_SUM()  
  
    def generate_SUM(self):  
        data = torch.randn(self.size, 3, dtype=torch.float32) * 2  
  
        label = data.prod(1).to(torch.float)  
  
        self.data = data  
        self.label = label  
  
    def __len__(self):  
        return self.size  
  
    def __getitem__(self, indx):  
        return self.data[indx], self.label[indx]  
  
  
train_data = SUMDataSet(20000)  
test_data = SUMDataSet(10000)  
# for i in range(len(train_data)):  
#     print(train_data[i])  
data_train_loader = data.DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True)  
loss_fn = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
  
  
def train_model(model, optimizer, data_loader, num_epochs=100):  
    model.train()  
    print(f"    |Обучение №{count}|")  
    for epoch in tqdm(range(num_epochs)):  
        for data_inputs, data_label in data_loader:  
            data_inputs = data_inputs.to(device)  
            data_label = data_label.to(device)  
  
            preds = model(data_inputs).squeeze(1)  
  
            loss = loss_fn(preds, data_label)  
  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
    model.eval()  
  
  
def test_model(model, data_loader):  
    model.eval()  
    total_accuracy, num_preds = 0.0, 0.0  
  
    with torch.no_grad():  
        for data_inputs, data_label in data_loader:  
            data_inputs, data_label = data_inputs.to(device), data_label.to(device)  
            preds = model(data_inputs).squeeze(1)  
  
            p = torch.abs(preds) + 1e-5  
            l = torch.abs(data_label) + 1e-5  
  
            accuracy_batch = torch.minimum(p, l) / torch.maximum(p, l)  
  
            total_accuracy += accuracy_batch.sum().item()  
            num_preds += data_label.shape[0]  
  
    mean_accuracy = total_accuracy / num_preds  
    print(f"\nAccuracy of the model: {100.0 * mean_accuracy:4.2f}%\n")  
  
data_test_loader = data.DataLoader(test_data, batch_size=512, shuffle=False, drop_last=True)  
test_model(model, data_test_loader)  
train_model(model, optimizer, data_train_loader)  
# for name, param in model.named_parameters():  
#     print(name, param)  
  
print("--- после обучения №1 ---")  
# test()  
count += 1  
test_model(model, data_test_loader)  
  
optimazer = torch.optim.Adam(model.parameters(), lr=0.001)  
train_model(model, optimazer, data_train_loader)  
print("--- после обучения №2 ---")  
# test()  
count += 1  
test_model(model, data_test_loader)  
  
optimazer = torch.optim.Adam(model.parameters(), lr=0.0001)  
data_train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)  
train_model(model, optimazer, data_train_loader, 50)  
print("--- после обучения №3 ---")  
test_model(model, data_test_loader)  
test()
```

**вот разница ReLU и GELU для моей модели:**

| Этап                          | ReLU   | GELU       | SiLU       |
| ----------------------------- | ------ | ---------- | ---------- |
| Без обучения                  | 17.88% | 16.16%     | 17.15%     |
| После №1 (lr=0.01, 100 эпох)  | 84.02% | 77.04%     | 84.79%     |
| После №2 (lr=0.001, 100 эпох) | 87.89% | **93.35%** | **94.63%** |
| После №3 (lr=0.0001, 50 эпох) | 88.87% | **94.56%** | **96.36%** |
