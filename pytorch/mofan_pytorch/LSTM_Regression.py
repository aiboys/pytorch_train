import torch 
import torch.nn as nn
from torch import optim
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


time_ = 10
input_size = 1
lr_rate = 0.1

steps = np.linspace(0, np.pi*2, 100, dtype = np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

plt.plot(steps, x_np, 'r-', label = 'input(sin)')
plt.plot(steps, y_np, 'b-', label = 'target(cos)')
plt.legend(loc= 'best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True)
        self.linear1 = nn.Linear(32, 1)

    def forward(self, x, state):
        # r_out: (batch_size, time_step, output_size)
        # h_state: (num_layers, batch, hidden_size)
        # x :      (batch_size, time_step, input_size)

        r_out, (h_state, c_state) = self.rnn(x, state)  # (batch_size, time_step, input_size)
        out = []
        #         print(h_state)
        #         print(h_state.shape)
        #         print(r_out.shape)  # 1 10 32
        # for time_step in range(r_out.size(1)):
        #      print("size:",r_out.size(1))   10
         #             out.append(self.linear1(r_out[:,time_step,:]))
        # #             print(np.array(out))
        #         print("stack",torch.stack(out, dim= 1).detach().cpu().data.numpy()) # 1 10 1
        out = self.linear1(r_out)
        #         print("c",out.cpu().data.numpy())
        return out, h_state, c_state


rnn = RNN().cuda()
print(rnn)

optimizer = optim.Adam(rnn.parameters(), lr = 0.02)
loss_function = nn.MSELoss()
h_state = torch.ones(1,1,32)
c_state = torch.ones(1,1,32)
plt.figure(1, figsize=(12, 5))
plt.ion()
for epoch in range(150):
    start, end = epoch*np.pi, (epoch+1)*np.pi
    steps = np.linspace(start, end, 10 , dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    x.requires_grad =True
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
#     h_state = Variable(h_state.data)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        h_state= h_state.cuda()
        c_state = c_state.cuda()
    losses = 0
    optimizer.zero_grad()
    out, h_state, c_state = rnn(x, (h_state, c_state))
    losses = loss_function(out, y)
    losses.backward(retain_graph=True)
    optimizer.step()

    print("epoch:{}  | loss:{}  ".format(epoch,losses))
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, out.cpu().data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)
    