import torch

def L2_normalize(X):
    norm_tensor = torch.norm(X, dim = 1).view(-1, 1).expand_as(X)
    X_norm = X.div(norm_tensor)

    return X_norm


def one_hot_tensor(X):
    ids = torch.LongTensor(X).view(-1, 1)
    n_row = X.shape[0]
    n_col = X.cat.categories.size
    Phi = torch.zeros(n_row, n_col)
    Phi.scatter_(dim = 1, index = ids, value = 1.)

    return Phi