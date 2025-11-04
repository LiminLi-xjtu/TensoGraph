import os
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from itertools import islice, combinations
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
from models.model import GTN,SemanticAttention
from models.layers import DEDICOMDecoder, weight_variable_glorot
import argparse



import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
'''
# Train on CPU (hide GPU) due to memory constraints
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.disable_v2_behavior()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
'''
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
# flags.DEFINE_integer('epochs',2000, 'Number of epochs to train.')
#flags.DEFINE_integer('embedding_dim', 256, 'Number of the dim of embedding')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('val_test_size', 0.1, 'the rate of validation and test samples.')
flags.DEFINE_float('beta',0.4, 'Loss function weight')
flags.DEFINE_float('beta1',0.3, 'Loss function weight')

parser = argparse.ArgumentParser()

parser.add_argument('--node_dim', type=int, default=32,
                        help='Node dimension')
parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
parser.add_argument('--epoch', type=int, default=500, help="n epoch")##如果是oneil数据为500，如果是oneil和cloud数据为200
parser.add_argument('--batch', type=int, default=256 ,help="batch size")
parser.add_argument('--gpu', type=int, default=1, help="cuda device")
parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
parser.add_argument('--suffix', type=str, default='results_oneil_mgaedc100_folds', help="model dir suffix")
parser.add_argument('--hidden', type=int, nargs='+', default=[2048, 4096, 8192], help="hidden size")
parser.add_argument('--lr', type=float, nargs='+', default=[1e-3, 1e-4, 1e-5], help="learning rate")
args = parser.parse_args()
node_dim = args.node_dim
num_channels = args.num_channels
num_layers = args.num_layers
norm = args.norm
adaptive_lr = args.adaptive_lr
args = parser.parse_args()
import os
import torch
import pickle
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score

from models.PRODeepSyn_utils import save_args, arg_min, conf_inv, calc_stat, save_best_model, find_best_model, random_split_indices
time_str = str(datetime.now().strftime('%y%m%d%H%M'))

OUTPUT= '/home/wangxi/HGTD-main/codes/result_dgt1/results_oneil_loewe/'
out_dir = os.path.join(OUTPUT, '{}'.format(args.suffix))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.isdir('/home/wangxi/HGTD-main/codes/result_dgt1'):
    os.makedirs('/home/wangxi/HGTD-main/codes/result_dgt1')

if not os.path.isdir('/home/wangxi/HGTD-main/codes/logs'):
    os.makedirs('/home/wangxi/HGTD-main/codes/logs')

log_file = os.path.join(out_dir, 'cv.log')
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=logging.INFO)

save_args(args, os.path.join(out_dir, 'args.json'))
test_loss_file = os.path.join(out_dir, 'test_loss.pkl')

if torch.cuda.is_available() and (args.gpu is not None):
    gpu_id = args.gpu
else:
    gpu_id = None
#some usefull funs
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()#形式转换为coo
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
    net_preds_binary = [ 1 if x >= 0.5 else 0 for x in net_preds ]
    net_auc = roc_auc_score(net_labels, net_preds)
    precision, recall, _ = precision_recall_curve(net_labels, net_preds)
    net_aupr = auc(recall, precision)
    net_acc = accuracy_score(net_preds_binary, net_labels)
    net_f1 = f1_score(net_labels, net_preds_binary)
    net_precision = precision_score(net_labels, net_preds_binary, zero_division=0)
    net_recall = recall_score(net_labels, net_preds_binary)
    fnoutput = [net_auc,net_acc, net_aupr, net_f1, net_precision, net_recall]
    return fnoutput
from models.PRODeepSyn_datasets import FastTensorDataLoader
# 1. load the data
import numpy as np
import pandas as pd
data = pd.read_csv('/media/imin/DATA/zhangdongxue/rawdata/oneil_loewe_cutoff30.txt', sep='\t', header=0)
data.columns = ['drugname1','drugname2','cell_line','synergy','fold']
drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2'])))) #38
drugscount = len(drugslist)
cellslist = sorted(list(set(data['cell_line'])))
cellscount = len(cellslist)

# get the features
# get the features
drug_feat = pd.read_csv('/media/imin/DATA/zhangdongxue/rawdata/oneil_drug_informax_feat.txt',sep='\t', header=None)
drug_feat = sp.csr_matrix( drug_feat )
drug_feat = sparse_to_tuple(drug_feat.tocoo())
num_drug_feat = drug_feat[2][1]
num_drug_nonzeros = drug_feat[1].shape[0]
# 读取txt文件
#node_features1 = np.loadtxt('/media/imin/DATA/zhangdongxue/process/oneil/oneil_drug_fingerprints.txt', dtype=int)
#node_features1= sp.csr_matrix(node_features1)
#node_features1=node_features1.todense()
node_features1=pd.read_csv('/media/imin/DATA/zhangdongxue/rawdata/oneil_drug_informax_feat.txt',sep='\t', header=None)
node_features1= sp.csr_matrix(node_features1)
node_features1=node_features1.todense()


resultspath = '/home/wangxi/HGTD-main/codes/result_dgt/results_oneil_loewe/embeddings'
if not os.path.isdir(resultspath):
    os.makedirs(resultspath)

#构造所有的药物组合为38*38
all_drug_indexs = []
for idx1 in range(drugscount):
    for idx2 in range(drugscount):
        drugname1 = drugslist[idx1]
        drugname2 = drugslist[idx2]
        all_drug_indexs.append(drugname1 + '&' +drugname2)
#构造所有的药物组合的下标组合和药物1下标小于药物2的下标的下标组合
indexs_all = []
indexs_all_triu = []
for idx1 in range(drugscount):
    for idx2 in range(drugscount):
        indexs_all.append([idx1, idx2])
        if idx1 < idx2:
            indexs_all_triu.append([idx1, idx2])
from torch.utils.data import Dataset
class FastSynergyDataset(Dataset):
    def __init__(self, drug_feat_file, embeddings_common, embeddings_specific,cell_feat_file, synergy_score_file, use_folds, train=True):
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
                        #drug1-drug2-cell
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
                            raw_sample = [drugslist.index(drug2), drugslist.index(drug1), cellslist.index(cellname), score]
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


##create model
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time

from tensorly.decomposition import parafac
from sklearn.preprocessing import Normalizer

from gem.embedding.static_graph_embedding  import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
import tensorly as tl
import numpy as np
from tensorly.decomposition import tucker
import pandas as pd
import networkx as nx
import torch
# 假设txt文件格式为：药物名称\tsmiles
df = pd.read_csv('/media/imin/DATA/zhangdongxue/rawdata/oneil_drug_smiles.txt', sep='\t', header=None,
                 names=['drug_name', 'smiles'])
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric import data as DATA
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


import numpy as np
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
compound_iso_smiles = []
compound_iso_smiles += list(df['smiles'])
class DrugGCN(torch.nn.Module):
    def __init__(self, num_atom_features):
        super(DrugGCN, self).__init__()
        self.conv1 = GCNConv(num_atom_features, num_atom_features)
        self.conv2 = GCNConv(num_atom_features, num_atom_features*2)
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
        x=torch.mean(x,dim=0)

        return x
class DNN(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, drug_feat3_len:int, cell_feat_len:int, hidden_size: int):
        super(DNN, self).__init__()

        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            #nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len),
        )

        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            #nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, drug_feat2_len),
        )


        self.drug_network3 = nn.Sequential(
            nn.Linear(drug_feat3_len, drug_feat3_len*2),
            nn.ReLU(),
            #nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(drug_feat3_len*2),
            nn.Linear(drug_feat3_len*2, drug_feat3_len),
        )


        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            #nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(cell_feat_len ),
            nn.Linear(cell_feat_len, 768),
        )

        self.fc_network = nn.Sequential(
            nn.Linear(2*(drug_feat1_len + drug_feat2_len + drug_feat3_len)+ 768, hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            #nn.Dropout(0.2),#如果是oneil数据不加dropout层，如果是oneil和cloud数据dropout为0.2
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug1_feat3: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, drug2_feat3: torch.Tensor, cell_feat: torch.Tensor):
        drug1_feat1_vector = self.drug_network1( drug1_feat1 )
        drug1_feat2_vector = self.drug_network2( drug1_feat2 )
        drug1_feat3_vector = self.drug_network3( drug1_feat3)
        drug2_feat1_vector = self.drug_network1( drug2_feat1 )
        drug2_feat2_vector = self.drug_network2( drug2_feat2 )
        drug2_feat3_vector = self.drug_network3( drug2_feat3 )
        cell_feat_vector = self.cell_network(cell_feat)
        # cell_feat_vector = cell_feat
        feat = torch.cat([drug1_feat1_vector, drug1_feat2_vector,drug1_feat3_vector , drug2_feat1_vector, drug2_feat2_vector, drug2_feat3_vector, cell_feat_vector], 1)
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

    def forward(self, drug1_feat1: torch.Tensor, drug2_feat1: torch.Tensor,  cell_feat: torch.Tensor):
        feat = torch.cat([drug1_feat1,drug2_feat1, cell_feat], 1)
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
        y_pred = model(drug1_feats1, drug1_feats2, drug1_feats3,drug2_feats1, drug2_feats2, drug2_feats3, cell_feats)
    else:
        yp1 = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2,drug2_feats3, cell_feats)
        yp2 = model(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2,drug1_feats3, cell_feats)
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
        self.q = nn.Linear(dim,dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim,dv)
    def forward(self,x):
        q = self.q(x)
        k = self.k(x)
        v=self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x =attn @ v
        return x
n_folds = 10
n_delimiter = 60
test_losses = []
test_pccs = []
class_stats = np.zeros((n_folds, 7))
from torch_geometric.nn import GCNConv
SYNERGY_FILE = '/media/imin/DATA/zhangdongxue/rawdata/oneil_loewe_cutoff30.txt'
DRUG_FEAT_FILE = '/media/imin/DATA/zhangdongxue/rawdata/oneil_drug_feat.npy'
CELL_FEAT_FILE = '/media/imin/DATA/zhangdongxue/rawdata/oneil_cell_feat.npy'
stats_loss = np.zeros((10,1))
# stats_auc = np.zeros((10,1))
for test_fold in range(10):
    # 划分训练集测试集验证集
    mdl_dir = os.path.join(out_dir, str(test_fold))
    if not os.path.exists(mdl_dir):
        os.makedirs(mdl_dir)

    valid_fold = list(range(10))[test_fold - 1]
    train_fold = [x for x in list(range(10)) if x != test_fold and x != valid_fold]
    print(train_fold, valid_fold, test_fold)
    test_data = data[data['fold'] == test_fold]  # test_data 2236*5   data 22737*5
    valid_data = data[data['fold'] == valid_fold]  # valid_data 2291*5
    train_data = data[(data['fold'] != test_fold) & (data['fold'] != valid_fold)]  # train_data 18210*5
    print('processing test fold {0} train folds {1} valid folds{2}.'.format(test_fold, train_fold, valid_fold))
    print('test shape{0} train shape{1} valid shape {2}'.format(test_data.shape, train_data.shape, valid_data.shape))

    # 在各个细胞系中划分训练集验证集测试集
    d_net1_norm_train = {}
    d_net2_norm_train = {}
    d_net3_norm_train = {}
    d_net1_norm_valid = {}
    d_net2_norm_valid = {}
    d_net3_norm_valid = {}
    d_net1_index_train = {}
    d_net2_index_train = {}
    d_net3_index_train = {}
    d_net1_index_valid = {}
    d_net2_index_valid = {}
    d_net3_index_valid = {}
    d_net1_labels_train = {}
    d_net2_labels_train = {}
    d_net3_labels_train = {}
    smile_graph = {}
    global_weights = {cellidx: weight_variable_glorot(node_dim, node_dim, name='weights_global') for cellidx in
                      range(cellscount)}
    local_weights0 = {
        cellidx: tf.reshape(weight_variable_glorot(node_dim, 1, name='weights_local_' + str(cellidx)), [-1]) for cellidx
        in range(cellscount)}
    local_weights1 = {
        cellidx: tf.reshape(weight_variable_glorot(node_dim, 1, name='weights_local_' + str(cellidx)), [-1]) for cellidx
        in range(cellscount)}
    local_weights2 = {
        cellidx: tf.reshape(weight_variable_glorot(node_dim, 1, name='weights_local_' + str(cellidx)), [-1]) for cellidx
        in range(cellscount)}


    def weight_variable_glorot2(input_dim, output_dim, name=""):
        initial = tf.random_uniform(
            [input_dim, output_dim],
            minval=0,
            maxval=1,
            dtype=tf.float32
        )  # 在[0,1]中间产生一个随机数
        return tf.Variable(initial, name=name)


    for smile in compound_iso_smiles:
        # print('smiles', smile)
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    smile_tensors = []
    for smiles in df['smiles']:
        c_size, features, edge_index = smile_graph[smiles]
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=torch.Tensor(features),
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            )
        drugGCN = DrugGCN(GCNData.x.shape[1])

        new_tensor = drugGCN(GCNData)
        new_tensor = new_tensor.detach()
        smile_tensors.append(new_tensor)
    smile_tensors = torch.stack(smile_tensors, dim=0)
    #node_features = smile_tensors
    embeddings_specific = {}
    for cellidx in range(cellscount):
        # cellidx = 0
        cellname = cellslist[cellidx]
        print('processing ', cellname)
        each_data = data[data['cell_line'] == cellname]
        net1_adj_mat_train = np.zeros((drugscount, drugscount))
        net2_adj_mat_train = np.zeros((drugscount, drugscount))
        net3_adj_mat_train = np.zeros((drugscount, drugscount))
        net1_adj_mat_valid = np.zeros((drugscount, drugscount))
        net2_adj_mat_valid = np.zeros((drugscount, drugscount))
        net3_adj_mat_valid = np.zeros((drugscount, drugscount))
        net1_adj_mat_test = np.zeros((drugscount, drugscount))
        net2_adj_mat_test = np.zeros((drugscount, drugscount))
        net3_adj_mat_test = np.zeros((drugscount, drugscount))
        net1_train_pos = []
        net2_train_pos = []
        net3_train_pos = []
        net1_valid_pos = []
        net2_valid_pos = []
        net3_valid_pos = []
        net1_test_pos = []
        net2_test_pos = []
        net3_test_pos = []

        for each in each_data.values:
            drugname1, drugname2, cell_line, synergy, fold = each
            drugidx1 = drugslist.index(drugname1)
            drugidx2 = drugslist.index(drugname2)
            if drugidx2 < drugidx1:
                drugidx1, drugidx2 = drugidx2, drugidx1
            # net1 协同作用
            if float(synergy) >= 30:
                # net1_pos.append([drugidx1, drugidx2])
                # train
                if fold in train_fold:  # train每个细胞系上协同作用的训练集
                    net1_adj_mat_train[drugidx1, drugidx2] = 1
                    net1_train_pos.append([drugidx1, drugidx2])
                    net1_train_pos.append([drugidx2, drugidx1])
                elif fold == valid_fold:  # valid每个细胞系上协同作用的验证集
                    net1_adj_mat_valid[drugidx1, drugidx2] = 1
                    net1_valid_pos.append([drugidx1, drugidx2])
                    net1_valid_pos.append([drugidx2, drugidx1])
                # #test
                elif fold == test_fold:  # test每个细胞系上协同作用的测试集
                    net1_adj_mat_test[drugidx1, drugidx2] = 1
                    net1_test_pos.append([drugidx1, drugidx2])
                    net1_test_pos.append([drugidx2, drugidx1])
            # net2加线作用
            elif (float(synergy) < 30) and (float(synergy) > -0):
                # net2_pos.append([drugidx1, drugidx2])
                # train
                if fold in train_fold:  # train每个细胞系上加线作用的训练集
                    net2_adj_mat_train[drugidx1, drugidx2] = 1
                    net2_train_pos.append([drugidx1, drugidx2])
                    net2_train_pos.append([drugidx2, drugidx1])
                elif fold == valid_fold:  # valid每个细胞系上加线作用的验证集
                    net2_adj_mat_valid[drugidx1, drugidx2] = 1
                    net2_valid_pos.append([drugidx1, drugidx2])
                    net2_valid_pos.append([drugidx2, drugidx1])
                # test
                elif fold == test_fold:  # test每个细胞系上加线作用的测试集
                    net2_adj_mat_test[drugidx1, drugidx2] = 1
                    net2_test_pos.append([drugidx1, drugidx2])
                    net2_test_pos.append([drugidx2, drugidx1])
            # net3
            elif float(synergy) < -0:
                # net3_pos.append([drugidx1, drugidx2])
                # train
                if fold in train_fold:  # train每个细胞系上对抗作用的训练集
                    net3_adj_mat_train[drugidx1, drugidx2] = 1
                    net3_train_pos.append([drugidx1, drugidx2])
                    net3_train_pos.append([drugidx2, drugidx1])
                elif fold == valid_fold:  # valid每个细胞系上对抗作用的验证集
                    net2_adj_mat_valid[drugidx1, drugidx2] = 1
                    net3_valid_pos.append([drugidx1, drugidx2])
                    net3_valid_pos.append([drugidx2, drugidx1])
                # test
                elif fold == test_fold:  # test每个细胞系上对抗作用的测试集
                    net2_adj_mat_test[drugidx1, drugidx2] = 1
                    net3_test_pos.append([drugidx1, drugidx2])
                    net3_test_pos.append([drugidx2, drugidx1])
        # net1
        net1_adj_mat_train1 = net1_adj_mat_train + net1_adj_mat_train.T
        net1_adj_mat_valid1 = net1_adj_mat_valid + net1_adj_mat_valid.T
        net1_adj_mat_test1 = net1_adj_mat_test + net1_adj_mat_test.T
        net1_adj_mat_train = sp.csr_matrix(net1_adj_mat_train)

        d_net1_norm_train[cellidx] = net1_adj_mat_train1

     # net2
        net2_adj_mat_train1 = net2_adj_mat_train + net2_adj_mat_train.T
        net2_adj_mat_valid1 = net2_adj_mat_valid + net2_adj_mat_valid.T
        net2_adj_mat_test1 = net2_adj_mat_test + net2_adj_mat_test.T

        d_net2_norm_train[cellidx] = net2_adj_mat_train1

        # net3
        net3_adj_mat_train1 = net3_adj_mat_train + net3_adj_mat_train.T
        net3_adj_mat_valid1 = net3_adj_mat_valid + net3_adj_mat_valid.T
        net3_adj_mat_test1 = net3_adj_mat_test + net3_adj_mat_test.T

        d_net3_norm_train[cellidx] = net3_adj_mat_train1

        import tensorly as tl
        num_classes=2
        md_array = np.zeros((drugscount, drugscount, 3))
        md_array[:, :, 0] = net1_adj_mat_train1
        md_array[:, :, 1] = net2_adj_mat_train1
        md_array[:, :, 2] = net3_adj_mat_train1
        XX = tl.tensor(md_array)
        rank=[3,3,3]
        core, factors = tucker(XX, rank)
        #node_features2 = torch.from_numpy(factors[0]).type(torch.FloatTensor)
        node_features = torch.from_numpy(node_features1).type(torch.FloatTensor)
        #core, factors = tucker(XX, rank)

        A = torch.from_numpy(net1_adj_mat_train1).type(torch.FloatTensor).unsqueeze(-1)
        A = torch.cat([A, torch.from_numpy(net2_adj_mat_train1).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
        A = torch.cat([A, torch.from_numpy(net3_adj_mat_train1).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
        A = torch.cat([A, torch.eye(drugscount).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)  # 添加一个单位对角阵
        for l in range(1):
            gtn = GTN(num_edge=A.shape[-1],  # edge类别的数量; 还有一个单位阵;
                        num_channels=num_channels,
                        w_in=node_features.shape[1],
                        w_out=node_dim,
                        num_class=num_classes,
                        num_layers=num_layers,  # GTLayer
                        norm=norm)
        embeddings_tem = gtn(A, node_features)
        sem = SemanticAttention(in_size=node_dim)
        embeddings_temp = sem(embeddings_tem)
        embeddings_temp1=torch.cat([torch.from_numpy(factors[0]).type(torch.FloatTensor), embeddings_temp],axis=1)
        #embeddings_temp1=torch.cat([embeddings_temp1, smile_tensors],axis=1)
        embeddings_specific[cellidx] = embeddings_temp1
    train_data = FastSynergyDataset(DRUG_FEAT_FILE, smile_tensors, embeddings_specific, CELL_FEAT_FILE,
                                    SYNERGY_FILE, use_folds=train_fold)
    valid_data = FastSynergyDataset(DRUG_FEAT_FILE, smile_tensors, embeddings_specific, CELL_FEAT_FILE,
                                    SYNERGY_FILE, use_folds=[valid_fold], train=False)
    test_data = FastSynergyDataset(DRUG_FEAT_FILE,smile_tensors, embeddings_specific, CELL_FEAT_FILE,
                                   SYNERGY_FILE, use_folds=[test_fold], train=False)
    logging.info("Outer: train folds {}, valid folds {} ,test folds {}".format(train_fold, valid_fold, test_fold))
    logging.info("-" * n_delimiter)

    train_loader = FastTensorDataLoader(*train_data.tensor_samples(), batch_size=args.batch, shuffle=True)
    valid_loader = FastTensorDataLoader(*valid_data.tensor_samples(), batch_size=len(valid_data))
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))
    best_hs, best_lr = args.hidden[2], args.lr[1]
    logging.info("Best hidden size: {} | Best learning rate: {}".format(best_hs, best_lr))
    model2= create_model(train_data, best_hs, gpu_id)
    # from models1.clr import cyclic_learning_rate
    #
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = cyclic_learning_rate(global_step=global_step, learning_rate=FLAGS.learning_rate * 0.1,
    #                                      max_lr=FLAGS.learning_rate, mode='exp_range', gamma=.995)
    optimizer1 = torch.optim.Adam(model2.parameters(), lr=best_lr)
    loss_func = nn.MSELoss(reduction='sum')
    min_loss = float('inf')
    train_loss = []
    valid_loss = []
    for epoch in range(1, args.epoch + 1):

        trn_loss = train_epoch(model2, train_loader, loss_func, optimizer1, gpu_id)
        # trn_loss /= train_loader.dataset_len
        # res2 = sess.run(reconstructions_specific, feed_dict=feed_dict)
        trn_loss /= train_loader.dataset_len

        val_loss = eval_epoch(model2, valid_loader, loss_func, gpu_id)
        val_loss /= valid_loader.dataset_len
        train_loss.append(trn_loss)
        valid_loss.append(val_loss)

        if epoch % 10 == 0:
            print("epoch: {} | train loss: {} valid loss {}".format(epoch, trn_loss, val_loss))
        if val_loss < min_loss:
            min_loss = val_loss
            save_best_model(model2.state_dict(), mdl_dir, epoch, keep=1)
    x1 = range(args.epoch)
    y1 = train_loss
    y2 = valid_loss
    plt.plot(x1, y1, 'g-', alpha=0.5, linewidth=1)
    plt.plot(x1, y2, 'r-', alpha=0.5, linewidth=1)
    plt.legend(['train_loss', 'valid_loss'], fontsize=10)
    plt.savefig(resultspath + str(test_fold) + '.png')
    plt.tight_layout()
    plt.show()

    model2.load_state_dict(torch.load(find_best_model(mdl_dir)), False)
    for cellidx in range(cellscount):
        embeddings = pd.DataFrame(embeddings_specific[cellidx].detach().numpy())
        embeddings.index = drugslist
        embeddings.to_csv(resultspath + '/results_embeddings_specific_' + str(cellidx) + '_' + str(test_fold) + '.txt',
                          sep='\t', header=None, index=True)
    with torch.no_grad():
        for test_each in test_loader:
            test_each = [x.cuda(args.gpu) for x in test_each]
            drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = test_each
            yp1 = model2(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats)
            yp2 = model2(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2, drug1_feats3, cell_feats)
            y_pred = (yp1 + yp2) / 2
            test_loss = loss_func(y_pred, y_true).item()
            y_pred = y_pred.cpu().numpy().flatten()
            y_true = y_true.cpu().numpy().flatten()
            test_pcc = np.corrcoef(y_pred, y_true)[0, 1]
            test_loss /= len(y_true)
            y_pred_binary = [1 if x >= 30 else 0 for x in y_pred]
            y_true_binary = [1 if x >= 30 else 0 for x in y_true]
            try:
                roc_score = roc_auc_score(y_true_binary, y_pred)
                precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
                auprc_score = auc(recall, precision)
                accuracy = accuracy_score(y_true_binary, y_pred_binary)
                f1 = f1_score(y_true_binary, y_pred_binary)
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary)
                kappa = cohen_kappa_score(y_true_binary, y_pred_binary)
                class_stat = [roc_score, auprc_score, accuracy, f1, precision, recall, kappa]
                class_stats[test_fold] = class_stat
            except ValueError:
                pass
            test_losses.append(test_loss)
            test_pccs.append(test_pcc)
            logging.info("Test loss: {:.4f}".format(test_loss))
            logging.info("Test pcc: {:.4f}".format(test_pcc))
            logging.info("*" * n_delimiter + '\n')

    ##cal the stats in each cell line mse
    from sklearn.metrics import mean_squared_error

    import warnings
    import pandas as pd
    from pandas.core.common import SettingWithCopyWarning

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

    all_data = pd.read_csv(SYNERGY_FILE, sep='\t', header=0)
    test_data_orig = all_data[all_data['fold'] == test_fold]
    test_data_orig['pred'] = y_pred
    test_data_orig.to_csv(out_dir + '/test_data_' + str(test_fold) + '.txt', sep='\t', header=True, index=False)
    cells_stats = np.zeros((cellscount, 9))
    for cellidx in range(cellscount):
        # cellidx = 0
        cellname = cellslist[cellidx]
        each_data = test_data_orig[test_data_orig['cell_line'] == cellname]
        each_true = each_data['synergy'].tolist()
        each_pred = each_data['pred'].tolist()
        each_loss = mean_squared_error(each_true, each_pred)
        each_pcc = np.corrcoef(each_pred, each_true)[0, 1]
        # class
        each_pred_binary = [1 if x >= 30 else 0 for x in each_pred]
        each_true_binary = [1 if x >= 30 else 0 for x in each_true]
        try:
            roc_score_each = roc_auc_score(each_true_binary, each_pred)  ## y_true=ground_truth
            precision, recall, _ = precision_recall_curve(each_true_binary, each_pred_binary)
            auprc_score_each = auc(recall, precision)
            accuracy_each = accuracy_score(each_true_binary, each_pred_binary)
            f1_each = f1_score(each_true_binary, each_pred_binary)
            precision_each = precision_score(each_true_binary, each_pred_binary, zero_division=0)
            recall_each = recall_score(each_true_binary, each_pred_binary)
            kappa_each = cohen_kappa_score(each_true_binary, each_pred_binary)
            t = [each_loss, each_pcc, roc_score_each, auprc_score_each, accuracy_each, f1_each, precision_each, recall_each,
             kappa_each]
            cells_stats[cellidx] = t
        except ValueError:
            pass

    pd.DataFrame(cells_stats).to_csv(out_dir + '/test_data_cells_stats_' + str(test_fold) + '.txt', sep='\t',
                                     header=None, index=None)

logging.info("CV completed")
with open(test_loss_file, 'wb') as f:
    pickle.dump(test_losses, f)
mu, sigma = calc_stat(test_losses)
logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
lo, hi = conf_inv(mu, sigma, len(test_losses))
logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
rmse_loss = [x ** 0.5 for x in test_losses]
mu, sigma = calc_stat(rmse_loss)
logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
pcc_mean, pcc_std = calc_stat(test_pccs)
logging.info("pcc: {:.4f} ± {:.4f}".format(pcc_mean, pcc_std))

class_stats = np.concatenate(
    [class_stats, class_stats.mean(axis=0, keepdims=True), class_stats.std(axis=0, keepdims=True)], axis=0)
pd.DataFrame(class_stats).to_csv(out_dir + '/class_stats.txt', sep='\t', header=None, index=None)






