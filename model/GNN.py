import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
from torch.autograd import Variable
 
class GNN(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(GNN, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number):
            self.encoder.append(DGCNLayer(opt, i))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]
        self.score_function1 = nn.Linear(opt["hidden_dim"]+opt["hidden_dim"],10)
        self.score_function2 = nn.Linear(10,1)
        self.MLP_ul = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()#opt["hidden_dim"])
        self.MLP_ul1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        self.MLP_vl = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()#opt["hidden_dim"])
        self.MLP_vl1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        
        self.mm2 = nn.Linear(opt["hidden_dim"] * opt["GNN"], opt["hidden_dim"]).cuda()#opt["hidden_dim"])
        self.mm3 = nn.Linear(opt["hidden_dim"] *opt["GNN"], opt["hidden_dim"]).cuda()
 
    def forward(self, ufea, vfea, UV_adj, VU_adj,adj):
        learn_user = ufea
        learn_item = vfea
        user = []
        item = []
        for i, layer in enumerate(self.encoder):
            if i % 2 == 1:
                # print("a",learn_user[0].shape)
                # learn_user[0] = F.dropout(learn_user[0], self.dropout, training=self.training)
                # learn_item[0] = F.dropout(learn_item[0], self.dropout, training=self.training)
                learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
                user.append(learn_user)
                item.append(learn_item)
            else:
                # print("b",learn_user.shape)
                # learn_user = F.dropout(learn_user, self.dropout, training=self.training)
                # learn_item = F.dropout(learn_item, self.dropout, training=self.training)
                learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
                user.append(learn_user[0])
                item.append(learn_item[0])
        self.user = user
        self.item = item
        m1 =nn.Softmax(dim=0)
 
        Hu = (torch.stack(user, dim=0)).mean(dim=1)#/user_layer[0].shape[0]
        Hv = (torch.stack(item, dim=0)).mean(dim=1)#/item_layer[0].shape[0]
        # Huc = (torch.stack(corr_user_layer, dim=0)).mean(dim=1)#/user_layer[0].shape[0]
        # Hvc = (torch.stack(corr_item_layer, dim=0)).mean(dim=1)#/item_layer[0].shape[0]
        
        alpha_ul = m1(self.MLP_ul1(nn.ReLU()(self.MLP_ul(Hu))))
        alpha_vl = m1(self.MLP_vl1(nn.ReLU()(self.MLP_vl(Hv))))
        # alpha_cul = m1(self.MLP_ul1(nn.ReLU()(self.MLP_ul(Huc))))
        # alpha_cvl = m1(self.MLP_vl1(nn.ReLU()(self.MLP_vl(Hvc))))
        # print( alpha_ul)
        alpha_ul1 = alpha_ul.repeat(1, user[0].shape[0]) #expand_as(torch.stack(user, dim=0)) 
        alpha_ul1 = alpha_ul1.unsqueeze(-1)
        
        alpha_vl1 = alpha_vl.repeat(1, item[0].shape[0]) #expand_as(torch.stack(user, dim=0)) 
        alpha_vl1 = alpha_vl1.unsqueeze(-1)
        # print( alpha_ul.shape)
        
        
        # user0 u01u02u03
        # h_u_final = (torch.stack(user, dim=0) * alpha_ul1).mean(dim=0) #/user_layer[0].shape[0
        # h_v_final = (torch.stack(item, dim=0) * alpha_vl1).mean(dim=0) #/user_layer[0].shape[0
        
#         hh = torch.stack(user, dim=0) * alpha_ul1
#         l=[]
#         for i in range(self.layer_number):

#             l.append(hh[i])

#         # print(torch.stack(hh, dim = 1))
        
    
    
#         hh1 = torch.stack(item, dim=0) * alpha_vl1
#         l1=[]
#         for i in range(self.layer_number):
#             l1.append(hh1[i])
        # print(torch.stack(hh, dim = 1))
        
        # print(torch.cat(l, dim = 1).shape)
        
        # h_u_final = F.relu(self.mm2(torch.cat(l, dim = 1)))#.sum(dim=0) #/user_layer[0].shape[0
        # h_v_final = F.relu(self.mm3(torch.cat(l1, dim = 1)))#.sum(dim=0) #/user_layer[0].shape[0
        # print(h_u_final.shape)
        # print("******************************************")
        h_u_final = (torch.stack(user, dim=0)*alpha_ul1).mean(dim=0).cuda()
        h_v_final = (torch.stack(item, dim=0)*alpha_vl1).mean(dim=0).cuda()
        
        # for i in range(10):
        #     print("#"*100)
        if self.layer_number % 2 == 0:
            return learn_user, learn_item, h_u_final, h_v_final, alpha_ul, alpha_vl, Hu, Hv
        return learn_user[0], learn_item[0], h_u_final, h_v_final, alpha_ul, alpha_vl, Hu, Hv
 
 
class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt,layer):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.layer = layer
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
 
        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
 
        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.dropout = opt["dropout"]
    
        self.user_union = nn.Linear(opt["feature_dim"] *2, opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] *2, opt["feature_dim"])
 
        self.user_union1 = nn.Linear(opt["feature_dim"] *3, opt["feature_dim"])
        self.item_union1 = nn.Linear(opt["feature_dim"] *3, opt["feature_dim"])
#         self.user_union = nn.Linear(opt["feature_dim"] , opt["feature_dim"])
#         self.item_union = nn.Linear(opt["feature_dim"] , opt["feature_dim"])
 
#         self.user_union1 = nn.Linear(opt["feature_dim"], opt["feature_dim"])
#         self.item_union1 = nn.Linear(opt["feature_dim"], opt["feature_dim"])
 
 
    def forward(self, ufea, vfea, UV_adj,VU_adj):
        if (self.layer % 2 == 0):
            User_n = self.gc1(vfea,UV_adj)
            Item_n = self.gc2(ufea,VU_adj)
            User_h = torch.cat((ufea,User_n),dim =1)
            Item_h = torch.cat((vfea,Item_n),dim =1)
            # User_h = (ufea+User_n)#/2
            # Item_h = (vfea+Item_n)#/2
            # print('1', User_h.shape)
 
            #### Reducing dim
            
            User_h = self.user_union(User_h)
            Item_h = self.item_union(Item_h)
 
            # User_h = F.dropout(User_h, self.dropout, training=self.training)
            # Item_h = F.dropout(Item_h, self.dropout, training=self.training)
        
            User_h = [F.relu(User_h) , User_n, ufea]
            Item_h = [F.relu(Item_h),Item_n, vfea]
            # User_h[0] = F.dropout(User_h[0], self.dropout, training=self.training)
            # Item_h[0]= F.dropout(Item_h[0], self.dropout, training=self.training)
 
            return User_h,Item_h
 
        elif (self.layer % 2 == 1):
            User_n1 = self.gc3(vfea[0],UV_adj) 
            User_n2 = self.gc1(vfea[1],UV_adj)
            Item_n1 = self.gc4(ufea[0],VU_adj) 
            Item_n2 = self.gc2(ufea[1],VU_adj)
            User_h = torch.cat((ufea[0],User_n2, User_n1),dim =1)
            Item_h = torch.cat((vfea[0],Item_n2, Item_n1),dim =1)
            
            
            # User_h = (ufea[0] + User_n2 + User_n1)#/3
            # Item_h = (vfea[0] + Item_n2 + Item_n1)#/3
 
            ### Reducing dim
            User_h = F.relu(self.user_union1(User_h)) #+ ufea[2]
            Item_h = F.relu(self.item_union1(Item_h)) #+ vfea[2]
            # User_h = F.dropout(User_h, self.dropout, training=self.training)
            # Item_h = F.dropout(Item_h, self.dropout, training=self.training)
 
            return User_h,Item_h