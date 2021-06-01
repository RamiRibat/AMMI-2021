from dlc_practical_prologue import *
from model import *

train_input, train_target,\
test_input, test_target = load_data(cifar = True,
                                    one_hot_labels = True,
                                    normalize = True,
                                    flatten = True)

# print(train_input.T.shape)
# print(train_target.T.shape)

ip_dim = train_input.T.shape[0]
op_dim = train_target.T.shape[0]
hid_un = [50] # Accept any shape
nEpochs = 10
lr = 0.01

mlp_model = MLP(ip_dim, op_dim, hid_un, Sig, dSig)
loss = mlp_model.train(train_input, train_target, nEpochs, lr, Loss, dLoss)
