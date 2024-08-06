import argparse
import numpy as np
import torch
from torch import nn
from utils import *
import torch.nn.functional as F
from layers import *
from loss import *
from torch.autograd import Variable
import torch.nn.functional as F
from meric import train_val_test_split
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import scipy.io as sio
import numpy as np


class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs, idcs_retrieval, idcs_query):
        self.dataset = dataset
        self.m_size = len(idxs)
        self.dataloader_train, _, _ = train_val_test_split(args, dataset, idxs, idcs_retrieval[id],idcs_query[id])
        self.device = args.device
        self.criterion_CL = ConLoss(temperature=0.07)
    def update_weights_UCCH_new(self, args, global_protos, global_avg_protos, model_img, private_modal):
        global_modal = model_img
        device = self.device
        loss_mse = nn.MSELoss().to(device)
        global_modal.to(device)
        private_modal.to(device)
        optimizer_global = torch.optim.Adam(global_modal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_private = torch.optim.Adam(private_modal.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


        private_modal.train()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(args.train_ep_private):
            for batch_idx, (img_F, txt_F, labels,index) in enumerate(self.dataloader_train):
                _, aff_norm, aff_label = affinity_tag_multi(labels.numpy(), labels.numpy())
                img = Variable(img_F.to(device).float())
                txt = Variable(torch.FloatTensor(txt_F.numpy()).to(device))
                labels = Variable(labels.to(device).float())
                labels = labels.to(device)
                fuseJ, H, pred = private_modal(img, txt)
                H_norm = F.normalize(H)
                clf_loss = loss_mse(torch.sigmoid(pred), labels)
                similarity_loss = loss_mse(H_norm.mm(H_norm.t()), torch.from_numpy(aff_label).float().to(device))
                B = torch.sign(H)
                lb = loss_mse(B, H)
                bal_loss = (torch.sum(B) / B.size(0))
                loss = similarity_loss * args.bb + lb * args.cc
                optimizer_private.zero_grad()
                loss.backward()
                clip_grad_norm_(private_modal.parameters(), 1.)
                optimizer_private.step()

        global_modal.train()
        private_modal.eval()

        for epoch in range(args.train_ep):
            for batch_idx, (img_F,txt_F,labels,index) in enumerate(self.dataloader_train):
                _, aff_norm, aff_label = affinity_tag_multi(labels.numpy(), labels.numpy())
                img = Variable(img_F.to(device))
                txt = Variable(torch.FloatTensor(txt_F.numpy()).to(device))
                labels = Variable(labels.to(device).float())

                bsz = labels.shape[0]

                fuseJg, Hg, predg = global_modal(img, txt)
                Hg_norm = F.normalize(Hg)
                clf_loss = loss_mse(torch.sigmoid(predg), labels)
                Lr = loss_mse(Hg_norm.mm(Hg_norm.t()), torch.from_numpy(aff_label).float().to(device))
                Bg = torch.sign(Hg)
                lb = loss_mse(Hg, Bg)
                bal_loss = (torch.sum(Bg) / Bg.size(0))
                loss = Lr * args.bb + lb * args.cc

                kd_loss = 0
                loss_cont = 0
                if len(global_protos)!=0:
                    fuseJp, private_modal_features, _ = private_modal(img, txt)
                    kd_loss = loss_mse(Hg, private_modal_features)
                    
                if len(global_protos) == args.num_users:
                    fuseJp, private_modal_features, _ = private_modal(img, txt)
                    features = fuseJp.unsqueeze(2)
                    for i in range(args.num_users):
                        for label in global_avg_protos.keys():
                            if label not in global_protos[i].keys():
                                global_protos[i][label] = global_avg_protos[label]
                        loss_cont += self.criterion_CL(features, labels, global_protos[i])

                loss = loss + kd_loss * args.ee + loss_cont * args.ff
                optimizer_global.zero_grad()
                loss.backward()
                clip_grad_norm_(global_modal.parameters(), 1.)
                optimizer_global.step()
                
        with torch.no_grad():
            private_modal.eval()
            private_modal.to(device)

            proj_bank_fuse = []
            n_samples = 0
            for bs, (images, texts, labels, indd) in enumerate(self.dataloader_train):
                img = Variable(images.to(self.device))
                txt = Variable(torch.FloatTensor(texts.numpy()).to(self.device))
                labels = Variable(labels.to(device).float())
                
                if n_samples >= self.m_size:
                    break
                fuseJ, H, _ = private_modal(img, txt)
                H = F.normalize(H)
                proj_bank_fuse.append(H)
                n_samples += len(indd)

            proj_bank_img = torch.cat(proj_bank_fuse, dim=0).contiguous()
            if n_samples > self.m_size:
                proj_bank_img = proj_bank_img[:self.m_size]

            local_protos = {}
            for i in range(args.classes):
                local_protos[i] = proj_bank_img[proj_bank_img.shape[0] // args.classes * i: proj_bank_img.shape[0] // args.classes * (i+1)].mean(0)
        return global_modal.cpu().state_dict(), private_modal.cpu().state_dict(), local_protos
    

class LocalTest(object):
    def __init__(self, args, id, dataset, idxs, idcs_retrieval,idcs_query):

        _, self.testloader, self.databaseloader = train_val_test_split(args, dataset, idxs, idcs_retrieval[id], idcs_query[id])
        self.device = args.device
    def test_and_eval(self, idx, args, local_model):
        device = args.device
        model = local_model
        model.to(device)
        model.eval()
        re_B = list([])
        re_L = list([])
        for _, (data_I, data_T, data_L, _) in enumerate(self.databaseloader):
            with torch.no_grad():
                var_data_I = Variable(data_I.to(device))
                var_data_T = Variable(torch.FloatTensor(data_T.numpy()).to(device))
                _, code, _ = model(var_data_I, var_data_T)
            code = torch.sign(code)
            re_B.extend(code.cpu().data.numpy())
            re_L.extend(data_L.cpu().data.numpy())
        qu_B = list([])
        qu_L = list([])
        for _, (data_I, data_T, data_L, _) in enumerate(self.testloader):
            with torch.no_grad():
                var_data_I = Variable(data_I.to(device))
                var_data_T = Variable(torch.FloatTensor(data_T.numpy()).to(device))
                _, code, _ = model(var_data_I, var_data_T)
            code = torch.sign(code)
            qu_B.extend(code.cpu().data.numpy())
            qu_L.extend(data_L.cpu().data.numpy())
        re_B = np.array(re_B)
        re_L = np.array(re_L)
        qu_B = np.array(qu_B)
        qu_L = np.array(qu_L)
        map = calculate_map(qu_B=qu_B, re_B=re_B, qu_L=qu_L, re_L=re_L)
        return map


                        
    
