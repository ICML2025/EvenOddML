import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN import GNN
from model.myDGI import myDGI
class EvenOddML(nn.Module):
    def __init__(self, opt):
        super(EvenOddML, self).__init__()
        self.opt=opt
        self.GNN = GNN(opt) # fast mode(GNN), slow mode(GNN2)
        self.DGI = myDGI(opt) # Since pytorch is not support sparse matrix well
        self.dropout = opt["dropout"]
        
        # degree one-hot features, Random 
        self.user_embedding = nn.Embedding(opt["number_user"], int(opt["feature_dim"]))
        self.item_embedding = nn.Embedding(opt["number_item"], int(opt["feature_dim"]))
        self.item_index = torch.arange(0, self.opt["number_item"], 1)
        self.user_index = torch.arange(0, self.opt["number_user"], 1)
        if self.opt["cuda"]:
            self.item_index = self.item_index.cuda()
            self.user_index = self.user_index.cuda()

    def score_predict(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(out.size()[0], -1)

    def score(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(-1)

    def forward(self, ufea, vfea, UV_adj, VU_adj, adj):
        learn_user,learn_item, h_u_final, h_v_final,alpha_ul, alpha_vl, Hu, Hv = self.GNN(ufea,vfea,UV_adj,VU_adj,adj)
        return learn_user,learn_item, h_u_final, h_v_final, self.GNN.user, self.GNN.item, alpha_ul, alpha_vl, Hu, Hv
