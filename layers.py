import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import math



class MLP(nn.Module):
    def __init__(self, hidden_dim=[1000, 2048, 512], act=nn.Tanh(), dropout=0.01):
        super(MLP, self).__init__()

        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim[-1]

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  

        # Add input layer
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
        self.activations.append(act)
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[0]))

        # Add hidden layers
        for i in range(len(self.hidden_dim) - 1):
            self.layers.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
            self.activations.append(act)
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim[i + 1])) 

        # Add output layer
        self.layers.append(nn.Linear(self.hidden_dim[-1], self.output_dim))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer, activation, batch_norm in zip(self.layers, self.activations, self.batch_norms):
            x = layer(x)
            x = activation(x)
            x = batch_norm(x)
            x = self.dropout(x)

        return x



class Fusion(nn.Module):
    def __init__(self, fusion_dim, nbit):
        super(Fusion, self).__init__()
        self.hash = nn.Sequential(
            nn.Linear(fusion_dim, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh()
        )

    def forward(self, x, y):
        hash_code = self.hash(x+y)
        return hash_code

class Layers(nn.Module):
    def __init__(self, args):
        super(Layers, self).__init__()
        self.image_dim = args.image_dim
        self.text_dim = args.text_dim

        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.bit)
        self.classes = args.classes
        self.batch_size = args.batch_size
        
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]
        self.fusionnn = Fusion(fusion_dim=self.common_dim, nbit=self.nbit)

        self.imageMLP = MLP(hidden_dim=self.img_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)
        self.textMLP = MLP(hidden_dim=self.txt_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)

        self.ifeat_gate = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim*2),
            nn.ReLU(),
            nn.Linear(self.common_dim*2, self.common_dim), 
            nn.Sigmoid())
        self.tfeat_gate = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim*2),
            nn.ReLU(),
            nn.Linear(self.common_dim*2, self.common_dim), 
            nn.Sigmoid())
        self.activation = nn.ReLU()
        self.neck = nn.Sequential(
            nn.Linear(self.common_dim,self.common_dim*4),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(self.common_dim*4,self.common_dim)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.nbit, self.common_dim),
            nn.ReLU()
        )

        self.hash_output = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.Tanh())
        self.classify = nn.Linear(self.nbit, self.classes)

    def forward(self, image, text, tgt=None):
        self.batch_size = len(image)
        imageH = self.imageMLP(image)#nbit length
        textH = self.textMLP(text)
        ifeat_info = self.ifeat_gate(imageH)#nbit length
        tfeat_info = self.tfeat_gate(textH)
        image_feat = ifeat_info*imageH
        text_feat = tfeat_info*textH
        fused_fine = self.fusionnn(image_feat, text_feat)
        cfeat_concat = self.fusion_layer(fused_fine)
        cfeat_concat = self.activation(cfeat_concat)
        nec_vec = self.neck(cfeat_concat)     
        code = self.hash_output(nec_vec)   
        return nec_vec, code, self.classify(code)


