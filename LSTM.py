from dataload import load_imdb
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as Data
from torchtext.vocab import GloVe
import os 
import matplotlib.pylab as plt

class LSTM(nn.Module):
    def __init__(self, vocab, embed_size=100, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<pad>'])
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hidden_size, 2)

        self._reset_parameters()
        
    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)
        _, (h_n, _) = self.rnn(x)
        output = self.fc(torch.cat((h_n[-1], h_n[-2]), dim=-1))
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class GRU(nn.Module):
    def __init__(self, vocab, embed_size=100, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size, padding_idx=vocab['<pad>'])
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(2 * hidden_size, 2)

        self._reset_parameters()
        
    def forward(self, x):
        x = self.embedding(x).transpose(0, 1)
        _, h_n = self.rnn(x)
        output = self.fc(torch.cat((h_n[-1], h_n[-2]), dim=-1))
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

device = torch.device('cuda')
print(device)
BATCH_SIZE=128
L_RATE=0.001
EPOCH=1
train_data, test_data, vocab=load_imdb()
#print(type(train_data))
# train_data=torch.from_numpy(train_data)
# test_data=torch.from_numpy(test_data)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0 )
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

#网络实例化
model = LSTM(vocab).to(device)
# print(model)
# os.system("pause")
#定义优化器
print(type(train_data))
optimizer = torch.optim.Adam(model.parameters(), L_RATE)
#定义交叉熵损失函数
loss_func = nn.CrossEntropyLoss().to(device)


train_loss = []
Acc = []
for epoch in range(20):
    model.train()
    running_loss = 0
    for batchidx, (x, label) in enumerate(train_loader):
        x, label = x.to(device), label.to(device)

        logits = model(x)
        loss = loss_func(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss.append(running_loss / len(train_loader))
    print('epoch:',epoch+1, 'loss:', loss.item())

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in test_loader:
            x, label = x.to(device),label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        Acc.append(total_correct / total_num)
    print(epoch, 'test acc:', acc)

plt.plot(Acc)
plt.title('Test Accuracy')
plt.ylabel('acc')
plt.xlabel('Step')
plt.show()

plt.plot(train_loss)
plt.title('Train Loss')
plt.ylabel('Loss')
plt.xlabel('Step')
plt.show()

torch.save(model, "F:/vs-written files/python/LSTM.pth")