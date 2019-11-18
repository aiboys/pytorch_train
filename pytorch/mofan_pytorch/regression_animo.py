import torch
import torch.nn.functional as F
import numpy
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.linspace(-1,1,100)
x = x.reshape(len(x),1)  # 变成2维度[[...]], torch里面只能处理至少两维的数
# x = torch.unsqueeze(x,dim=1)  # 这个也可以多一个维度
# x
y = x.pow(2) + 0.2*torch.rand(x.size())  #加上一些噪声
x, y = Variable(x), Variable(y)
# fig = plt.figure(1, figsize=(6,6))
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, 10)
        self.predict = nn.Linear(10, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
if torch.cuda.is_available():
    net.cuda()
print(net)


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
net.apply(init_weight)


optimizer = optim.SGD(net.parameters(), lr = 0.3)
loss = nn.MSELoss()

plt.ion()  # 实时打印
plt.show()

for epoch in range(20):
    losses = 0
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    optimizer.zero_grad()
    prediction = net(x)
    losses = loss(prediction, y)
    losses.backward()
    optimizer.step()
    print("epoch:{},\t loss:{}".format(epoch, losses))
    if epoch%5==0:
        plt.cla()
        plt.scatter(x.cpu().data.numpy(),y.cpu().data.numpy())
        plt.plot(x.cpu().data.numpy(), prediction.cpu().data.numpy(),'r-', lw=3)
        plt.text(0.5,0,'loss=%.3f'% losses, fontdict={'size':20,'color': 'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()


