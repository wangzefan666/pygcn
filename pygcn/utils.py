import numpy as np
import scipy.sparse as sp
import torch


def class2label(labels):
    """
    param:
        labels - np.array, list of items' classes in string form
    """
    classes = list(set(labels))
    class_dict = {classes[i]: i for i in range(len(classes))}
    # 技巧：用字典对 list(array) 类型进行转化
    labels_ind = np.array(list(map(class_dict.get, labels)), dtype=np.int32)
    return labels_ind


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.str)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 技巧：可以将 array 直接转化成 sparse matrix，注意类型，否则没有意义
    features = sp.csr_matrix(idx_features_labels[:, 1:-1].astype(np.float32))
    labels = class2label(idx_features_labels[:, -1])

    # build graph
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    # 注意，这里adj不是半角矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # norm
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # all have 2708 items
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # convert to tensor
    features = torch.FloatTensor(np.array(features.todense()))  # todense()返回的是matrix
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    Pay attention to data type.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
