import torch


def get_batch_codes(batch_mat, batch_key):
    if type(batch_key) is str or len(batch_key) == 1:
        if not type(batch_key) is str:
            batch_key = batch_key[0]

        batch_vec = batch_mat[batch_key]

    else:
        df = batch_mat[batch_key].astype('str')
        batch_vec = df.apply(lambda row: ','.join(row), axis = 1)
    
    return batch_vec.astype("category").cat.codes.astype("category")


def one_hot_tensor(X, device_type):
    ids = torch.tensor(X).long().to(device_type).view(-1, 1)
    n_row = X.shape[0]
    n_col = X.cat.categories.size
    Phi = torch.zeros(n_row, n_col, dtype=torch.float, device = device_type)
    Phi.scatter_(dim=1, index=ids, value=1.0)

    return Phi
