import torch as th
from functions import *


class MLP:
    def __init__(self, ip_dim, op_dim, hid_un, Sig, dSig, seed=0):
        print('Initialize MLP Model!')
        self.L = len(hid_un)
        self.Sig = Sig
        self.dSig = dSig
        self.w, self.b, self.X, self.S = [], [], [], []

        current_ip = ip_dim
        for i in range(len(hid_un)):
            w = th.rand(hid_un[i], current_ip, requires_grad=True) * 1e-6
            b = th.rand(hid_un[i], 1, requires_grad=True) * 1e-6
            self.w.append(w); self.b.append(b)
            # S = lambda x, w, b: w.mm(x) + b
            S = forward
            X = Sig
            self.S.append(S); self.X.append(X)
            current_ip = hid_un[i]

        w = th.rand(op_dim, hid_un[-1], requires_grad=True) * 1e-6
        b = th.rand(op_dim, 1, requires_grad=True) * 1e-6
        self.w.append(w); self.b.append(b)
        S = forward
        X = Sig
        self.S.append(S); self.X.append(X)


    def train(self, Xtrain, Ytrain, nEpochs, lr, Loss, dLoss):
        for ep in range(nEpochs):
            X, Y = Xtrain.T, Ytrain.T
            dL_dw = [th.zeros(w.shape[0], w.shape[1]) for w in self.w]
            print('dL_dw[0] ', dL_dw[0].shape)
            dL_db = [th.zeros(b.shape[0], b.shape[1]) for b in self.b]

            inds = th.randperm(len(Xtrain))
            for inx in inds:
                x, y = X[:, inx].view(-1,1), Y[:, inx].view(-1,1)
                S_funs, X_funs = self.forward_pass(x)
                y_hat = X_funs[-1]
                dL = dLoss(y_hat, y)
                dL_dw, dL_db = self.backward_pass(y, x, dL_dw, dL_db, dL)

            print('len(dL_dw) ', len(dL_dw))
            print('len(self.w) ', len(self.w))
            for j in range(len(dL_dw)):
                self.w[j] -= lr*dL_dw[j]
                self.b[j] -= lr*dL_db[j]

            S_funs, X_funs = self.forward_pass(X)
            Y_hat = X_funs[-1]
            # print(Y_hat.shape)
            error = Loss(Y_hat, Y)
            print(f'Epoch: {ep}, Error = {error}', end='\r')
        loss = error
        return loss

    def test(self, Xtest, Ytest, Acc):
        pass


    def forward_pass(self, x):
        # print('ForwPass')
        S_funs = []
        X_funs = []
        X_funs.append(x)
        # print('len(self.X): ', len(self.X))
        for i in range(len(self.w)):
            S = self.S[i](X_funs[i], self.w[i], self.b[i])
            # print(f'S{i} shape: {S.shape}')
            X = self.X[i](S)
            # print(f'X{i} shape: {X.shape}')
            S_funs.append(S); X_funs.append(X)
        # print('X_funs: ', X_funs)
        return S_funs, X_funs



    def backward_pass(self, t, x, dL_dw, dL_db, dL):
        # dL_dw, dL_db = [], []
        Xi = [x]
        for i in range(len(self.w)):
            Xi.append(self.X[i](self.S[i](Xi[i], self.w[i], self.b[i])))
        for l in range(1, len(self.w)):
            dw = self.dLdS(l, x, dL).mm(Xi[l-1].T)
            db = self.dLdS(l, x, dL)
            dL_dw[l-1] += dw
            dL_db[l-1] += db
        return dL_dw, dL_db


    def dLdS(self, l, x, dL):
        if l < self.L:
            dl_dx = (self.w[l-1].T).mm(self.dLdS(l+1, x))
            return dl_dx*self.dSig(self.S[l-1](x, self.w[l-1], self.b[l-1]))
        else:
            return dL*self.dSig(self.S[l-1](x, self.w[l-1], self.b[l-1]))
