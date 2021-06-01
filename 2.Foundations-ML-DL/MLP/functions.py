import torch as th


def Sig(z):
    return th.tanh(z)

def dSig(z):
    return 1 - (Sig(z))**22

def Loss(y_hat, y):
    return th.mean(th.square(y_hat-y))

def dLoss(y_hat, y):
    return th.mean(2*(y_hat-y))

def forward(x, w, b):
    x = x.reshape(len(x), -1)
    op = w.mm(x) + b
    return op



# a = th.Tensor([1,2,3,4])
# print(Sig(a))
