import torch
import torch.nn as nn   
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data.dataset import Dataset
import copy
import os
import random
import logging
import os.path as osp
import scipy.io as sio

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def calculate_topk(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def getdataset(args, data_path):
    with h5py.File(data_path, 'r') as file:
        train_X = np.array(file['I_tr']).T.astype(np.float32)
        train_Y = np.array(file['T_tr']).T.astype(np.float32)
        train_L = np.array(file['L_tr']).T.astype(int)
        query_X = np.array(file['I_te']).T.astype(np.float32)
        query_Y = np.array(file['T_te']).T.astype(np.float32)
        query_L = np.array(file['L_te']).T.astype(int)
        retrieval_X = np.array(file['I_db']).T.astype(np.float32)
        retrieval_Y = np.array(file['T_db']).T.astype(np.float32)
        retrieval_L = np.array(file['L_db']).T.astype(int)
    return train_L, train_X, train_Y, retrieval_L, retrieval_X, retrieval_Y, query_L, query_X, query_Y



def prepare_data_noniid(args):
    if args.dataset == 'flickr':
        datapath = "data/mir.h5"
    elif args.dataset == 'wiki':
        datapath = "data/wiki.h5"
    elif args.dataset == 'nuswide':
        datapath = "data/nus.h5"

    train_L, train_x, train_y, retrieval_L, retrieval_x, retrieval_y, query_L, query_x, query_y = getdataset(args, datapath)
    n_clients = args.num_users
    alpha = 0.1
    datasetall = [train_L, train_x, train_y, retrieval_L, retrieval_x, retrieval_y, query_L, query_x, query_y]

    n_classes = train_L.shape[1]
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    client_idcs_train = [[] for _ in range(n_clients)]
    client_idcs_retrieval = [[] for _ in range(n_clients)]
    client_idcs_query = [[] for _ in range(n_clients)]

    for i, label in enumerate(train_L):
        client_idx = np.random.choice(n_clients, p=label_distribution[np.argmax(label)])
        client_idcs_train[client_idx].append(i)

    for i, label in enumerate(retrieval_L):
        client_idx = np.random.choice(n_clients, p=label_distribution[np.argmax(label)])
        client_idcs_retrieval[client_idx].append(i)

    for i, label in enumerate(query_L):
        client_idx = np.random.choice(n_clients, p=label_distribution[np.argmax(label)])
        client_idcs_query[client_idx].append(i)

    return datasetall, client_idcs_train, client_idcs_retrieval, client_idcs_query

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def update_protos(local_protos, proj_bank_img, alpha=0.5, threshold=0.1):
    class_counts = {}
    for i in range(proj_bank_img.shape[0]):
        for label in range(len(local_protos)):
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

    updated_protos = {}
    for class_label in class_counts:
        class_features = proj_bank_img[class_counts[class_label] * class_label : class_counts[class_label] * (class_label + 1)]
        new_proto = class_features.mean(0)
        if class_label in local_protos:
            updated_protos[class_label] = alpha * local_protos[class_label] + (1 - alpha) * new_proto
        else:
            updated_protos[class_label] = new_proto

    for class_label in list(updated_protos.keys()):
        distances = torch.norm(updated_protos[class_label] - proj_bank_img, dim=1)
        if distances.min() > threshold:
            del updated_protos[class_label]

    for class_label in range(len(local_protos)):
        if class_label not in updated_protos:
            updated_protos[class_label] = proj_bank_img[class_counts[class_label] * class_label : class_counts[class_label] * (class_label + 1)].mean(0)

    return updated_protos

def average_weights(w):

    w_avg = copy.deepcopy(w)

    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.div(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg


def zero2eps(x):

    x[x == 0] = 1
    return x


def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum
    in_affnty = np.transpose(affinity/row_sum)
    return in_affnty, out_affnty


def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)
    return in_aff, out_aff, affinity_matrix

def global_guidance_loss(local_protos, global_protos):
    loss = 0
    for label, local_proto in local_protos.items():
        if label in global_protos:
            global_proto = global_protos[label].to(local_proto.device)
            loss += F.mse_loss(local_proto, global_proto)
    return loss


def js_loss(x1, xa, t=0.1, t2=0.01):
    xa = xa.unsqueeze(0)  
    xa = xa.expand(x1.size(0), -1)  
    pred_sim = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())
    inputs = F.log_softmax(pred_sim / t, dim=1)
    target_js = F.softmax(pred_sim / t2, dim=1)
    js_loss = F.kl_div(inputs, target_js, reduction="batchmean")
    return js_loss


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):

    num_query = qu_L.shape[0]
    topkmap = 0
    maps = []
    ids = []
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
        maps.append(topkmap_)
        ids.append(iter)
    topkmap = topkmap / num_query
    return topkmap

def calculate_hamming(B1, B2):

    leng = B2.shape[1]
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(self.images) == len(self.labels)
        return count



def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def logger(fileName='log'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    log_name = str(fileName) + '.txt'
    log_dir = './logs'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)
    return logger

