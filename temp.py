import torch
print('in')
temp = torch.randn((16, 10000))
print(torch.mean(torch.var(temp, 1)))