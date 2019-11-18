import torch
from torch.autograd import  Variable
import numpy as np
a = np.array([[1.,2.],[3.,4.]])
aa = torch.tensor(a, requires_grad =True)
aaa = torch.Tensor(a)
# aaa.requires_grad_()
aaa.requires_grad=True
b = torch.FloatTensor([[1,2],[3,4]])
c = Variable(b, requires_grad = True)
print(aa)
print(aaa)
print(b)
print(c)

