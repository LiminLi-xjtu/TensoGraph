from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, \
    accuracy_score, precision_score, recall_score, cohen_kappa_score

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv




def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()  # 形式转换为coo
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def get_classification_stats(net_labels, net_preds):
    net_preds_binary = [1 if x >= 0.5 else 0 for x in net_preds]
    net_auc = roc_auc_score(net_labels, net_preds)
    precision, recall, _ = precision_recall_curve(net_labels, net_preds)
    net_aupr = auc(recall, precision)
    net_acc = accuracy_score(net_preds_binary, net_labels)
    net_f1 = f1_score(net_labels, net_preds_binary)
    net_precision = precision_score(net_labels, net_preds_binary, zero_division=0)
    net_recall = recall_score(net_labels, net_preds_binary)
    fnoutput = [net_auc, net_acc, net_aupr, net_f1, net_precision, net_recall]
    return fnoutput




def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index



def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()  ##一个药物字符串中原子的数量

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  ##78个原子特征
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index





class DrugGCN(torch.nn.Module):
    def __init__(self, num_atom_features):
        super(DrugGCN, self).__init__()
        self.conv1 = GCNConv(num_atom_features, num_atom_features)
        self.conv2 = GCNConv(num_atom_features, num_atom_features * 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        # 第一层GCN
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=0)

        return x


class DNN(nn.Module):
    def __init__(self, drug_feat1_len: int, drug_feat2_len: int, drug_feat3_len: int, cell_feat_len: int,
                 hidden_size: int):
        super(DNN, self).__init__()

        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len * 2),
            nn.ReLU(),
            # nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(drug_feat1_len * 2),
            nn.Linear(drug_feat1_len * 2, drug_feat1_len),
        )

        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len * 2),
            nn.ReLU(),
            # nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(drug_feat2_len * 2),
            nn.Linear(drug_feat2_len * 2, drug_feat2_len),
        )

        self.drug_network3 = nn.Sequential(
            nn.Linear(drug_feat3_len, drug_feat3_len * 2),
            nn.ReLU(),
            # nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(drug_feat3_len * 2),
            nn.Linear(drug_feat3_len * 2, drug_feat3_len),
        )

        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            # nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(cell_feat_len),
            nn.Linear(cell_feat_len, 768),
        )

        self.fc_network = nn.Sequential(
            nn.Linear(2 * (drug_feat1_len + drug_feat2_len + drug_feat3_len) + 768, hidden_size),
            nn.ReLU(),
            # nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug1_feat3: torch.Tensor,
                drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, drug2_feat3: torch.Tensor,
                cell_feat: torch.Tensor):
        drug1_feat1_vector = self.drug_network1(drug1_feat1)
        drug1_feat2_vector = self.drug_network2(drug1_feat2)
        drug1_feat3_vector = self.drug_network3(drug1_feat3)
        drug2_feat1_vector = self.drug_network1(drug2_feat1)
        drug2_feat2_vector = self.drug_network2(drug2_feat2)
        drug2_feat3_vector = self.drug_network3(drug2_feat3)
        cell_feat_vector = self.cell_network(cell_feat)
        # cell_feat_vector = cell_feat
        feat = torch.cat(
            [drug1_feat1_vector, drug1_feat2_vector, drug1_feat3_vector, drug2_feat1_vector, drug2_feat2_vector,
             drug2_feat3_vector, cell_feat_vector], 1)
        out = self.fc_network(feat)
        return out


class DNN_orig(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN_orig, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug2_feat1: torch.Tensor, cell_feat: torch.Tensor):
        feat = torch.cat([drug1_feat1, drug2_feat1, cell_feat], 1)
        out = self.network(feat)
        return out


def create_model(data, hidden_size, gpu_id=None):
    # model = DNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    model = DNN(data.drug_feat1_len(), data.drug_feat2_len(), data.drug_feat3_len(), data.cell_feat_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model


def step_batch(model, batch, loss_func, gpu_id=None, train=True):
    if gpu_id is not None:
        batch = [x.cuda(gpu_id) for x in batch]
    drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = batch
    # if gpu_id is not None:
    # drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = drug1_feats1.cuda(gpu_id), drug1_feats2.cuda(gpu_id),drug1_feats3.cuda(gpu_id), drug2_feats1.cuda(gpu_id), drug2_feats2.cuda(gpu_id), drug2_feats3.cuda(gpu_id), cell_feats.cuda(gpu_id), y_true.cuda(gpu_id)
    if train:
        y_pred = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats)
    else:
        yp1 = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats)
        yp2 = model(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2, drug1_feats3, cell_feats)
        y_pred = (yp1 + yp2) / 2
    loss = loss_func(y_pred, y_true)
    return loss


'''
def train_epoch(model, loader, loss_func, optimizer, lambda_reg=0.001, gpu_id=None):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = step_batch(model, batch, loss_func, gpu_id)

        # Calculate L2 regularization
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param)

        # Add L2 regularization to the loss
        loss += lambda_reg * l2_reg.cuda(gpu_id)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss
'''


def train_epoch(model, loader, loss_func, optimizer, gpu_id=None):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = step_batch(model, batch, loss_func, gpu_id)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def eval_epoch(model, loader, loss_func, gpu_id=None):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch in loader:
            loss = step_batch(model, batch, loss_func, gpu_id, train=False)
            epoch_loss += loss.item()
    return epoch_loss


class Self_Attention(nn.Module):
    def __init__(self, dim, dk, dv):
        super(Self_Attention, self).__init__()
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        return x



class FastSynergyDataset(Dataset):
    def __init__(self, drugslist, cellslist, drug_feat_file, embeddings_common, embeddings_specific, cell_feat_file, synergy_score_file,
                 use_folds, train=True):
        # self.drug2id = read_map(drug2id_file)
        # self.cell2id = read_map(cell2id_file)
        self.drug_feat1 = np.load(drug_feat_file)
        self.drug_feat2 = embeddings_common
        self.drug_feat3 = embeddings_specific
        self.cell_feat = np.load(cell_feat_file)
        self.samples = []
        self.raw_samples = []
        self.train = train
        valid_drugs = set(drugslist)
        valid_cells = set(cellslist)
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(float(fold)) in use_folds:
                        # drug1-drug2-cell
                        sample = [
                            torch.from_numpy(self.drug_feat1[drugslist.index(drug1)]).float(),
                            (self.drug_feat2[drugslist.index(drug1)]).float(),
                            (self.drug_feat3[cellslist.index(cellname)][drugslist.index(drug1)]).float(),

                            torch.from_numpy(self.drug_feat1[drugslist.index(drug2)]).float(),
                            (self.drug_feat2[drugslist.index(drug2)]).float(),
                            (self.drug_feat3[cellslist.index(cellname)][drugslist.index(drug2)]).float(),
                            torch.from_numpy(self.cell_feat[cellslist.index(cellname)]).float(),
                            torch.FloatTensor([float(score)]),
                        ]
                        self.samples.append(sample)
                        raw_sample = [drugslist.index(drug1), drugslist.index(drug2), cellslist.index(cellname), score]
                        self.raw_samples.append(raw_sample)
                        if train:
                            ###drug2-drug1-cell
                            sample = [
                                torch.from_numpy(self.drug_feat1[drugslist.index(drug2)]).float(),
                                (self.drug_feat2[drugslist.index(drug2)]).float(),
                                (self.drug_feat3[cellslist.index(cellname)][drugslist.index(drug2)]).float(),
                                torch.from_numpy(self.drug_feat1[drugslist.index(drug1)]).float(),
                                (self.drug_feat2[drugslist.index(drug1)]).float(),
                                (self.drug_feat3[cellslist.index(cellname)][drugslist.index(drug1)]).float(),
                                torch.from_numpy(self.cell_feat[cellslist.index(cellname)]).float(),
                                torch.FloatTensor([float(score)]),
                            ]

                            self.samples.append(sample)
                            raw_sample = [drugslist.index(drug2), drugslist.index(drug1), cellslist.index(cellname),
                                          score]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat1_len(self):
        return self.drug_feat1.shape[-1]

    def drug_feat2_len(self):
        return self.drug_feat2.shape[-1]

    def drug_feat3_len(self):
        return 35

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1_f1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d1_f2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        d1_f3 = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        d2_f1 = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        d2_f2 = torch.cat([torch.unsqueeze(self.samples[i][4], 0) for i in indices], dim=0)
        d2_f3 = torch.cat([torch.unsqueeze(self.samples[i][5], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][6], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][7], 0) for i in indices], dim=0)
        return d1_f1, d1_f2, d1_f3, d2_f1, d2_f2, d2_f3, c, y




def weight_variable_glorot2(input_dim, output_dim, name=""):
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=0,
        maxval=1,
        dtype=tf.float32
    )  # 在[0,1]中间产生一个随机数
    return tf.Variable(initial, name=name)