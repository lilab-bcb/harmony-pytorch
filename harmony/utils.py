import torch

def one_hot_tensor(X):
    ids = torch.LongTensor(X).view(-1, 1)
    n_row = X.shape[0]
    n_col = X.cat.categories.size
    Phi = torch.zeros(n_row, n_col)
    Phi.scatter_(dim = 1, index = ids, value = 1.)

    return Phi