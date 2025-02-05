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
        
        self.MLP_ul1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        # self.MLP_ul2 = nn.Linear(opt["hidden_dim"], 1).cuda()
        # self.MLP_ul3 = nn.Linear(opt["hidden_dim"], 1).cuda()
        # self.MLP_ul4 = nn.Linear(opt["hidden_dim"], 1).cuda()
        self.MLP_ul = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()#opt["hidden_dim"])
        self.MLP_vl = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()#opt["hidden_dim"])
        
        self.MLP_vl1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        # self.MLP_vl2 = nn.Linear(opt["hidden_dim"], 1).cuda()
        # self.MLP_vl3 = nn.Linear(opt["hidden_dim"], 1).cuda()
        # self.MLP_vl4 = nn.Linear(opt["hidden_dim"], 1).cuda()
        
    def forward1(self, ufea, vfea, UV_adj, VU_adj,adj):
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
        
        # print(len(user), len(user[0]))
        alpha_u = []
        alpha_v = []
        alpha_u = m1(self.MLP_ul1(torch.stack(user, dim=0)))
        alpha_v = m1(self.MLP_vl1(torch.stack(item, dim=0)))
        # alpha_u = alpha_u.repeat(1, 1, 128)
        # alpha_v = alpha_v.repeat(1, 1, 128)
        
        # print(alpha_u.shape)
        # print(torch.stack(user, dim=0).shape)
        Hu = (torch.mul(torch.stack(user, dim=0), alpha_u)).sum(dim=0)#/user_layer[0].shape[0]
        Hv = (torch.mul(torch.stack(item, dim=0), alpha_v)).sum(dim=0)#/item_layer[0].shape[0]
        # print(Hu.shape)
        # Hu : (4, 128)
        
        # Huc = (torch.stack(corr_user_layer, dim=0)).mean(dim=1)#/user_layer[0].shape[0]
        # Hvc = (torch.stack(corr_item_layer, dim=0)).mean(dim=1)#/item_layer[0].shape[0]
        
        # layer attention
        alpha_ul = m1(self.MLP_ul(Hu)).squeeze(1)
        alpha_vl = m1(self.MLP_vl(Hv)).squeeze(1)
        
        # alpha_cul = m1(self.MLP_ul1(nn.ReLU()(self.MLP_ul(Huc))))
        # alpha_cvl = m1(self.MLP_vl1(nn.ReLU()(self.MLP_vl(Hvc))))
        # print( alpha_ul)
        
        # alpha_ul1 = alpha_ul.repeat(1, user[0].shape[0]) #expand_as(torch.stack(user, dim=0)) 
        # alpha_ul1 = alpha_ul1.unsqueeze(-1)
        # print(alpha_ul1.shape)
        # alpha_vl1 = alpha_vl.repeat(1, item[0].shape[0]) #expand_as(torch.stack(user, dim=0)) 
        # alpha_vl1 = alpha_vl1.unsqueeze(-1)
        # print( alpha_ul.shape)
        
        # user0 u01u02u03
        # print(alpha_ul.shape)
        # print(torch.stack(user, dim=0).shape)
        # h_u_final = (torch.stack(user, dim=0) * alpha_ul).sum(dim=0)
        h_u_final = (torch.mul(torch.stack(user, dim=0), alpha_ul)).mean(dim=0)
        h_v_final = (torch.mul(torch.stack(item, dim=0), alpha_vl)).mean(dim=0)
        
        # h_u_final = (torch.stack(user, dim=0) * alpha_ul).sum(dim=0) #/user_layer[0].shape[0
        # h_v_final = (torch.stack(item, dim=0) * alpha_vl).sum(dim=0) #/user_layer[0].shape[0
        # exit()
        # h_u_final = (torch.stack(user, dim=0)).mean(dim=0).cuda()
        # h_v_final = (torch.stack(item, dim=0)).mean(dim=0).cuda()
        
        # for i in range(10):
        #     print("#"*100)
        if self.layer_number % 2 == 0:
            return learn_user, learn_item, h_u_final, h_v_final, alpha_ul.squeeze(1), alpha_vl.squeeze(1), Hu, Hv
        return learn_user[0], learn_item[0], h_u_final, h_v_final, alpha_ul.squeeze(1), alpha_vl.squeeze(1), Hu, Hv

    ######## Approach 2
    
    
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
        Hu = (torch.stack(user, dim=0)).sum(dim=1)#/user_layer[0].shape[0]
        Hv = (torch.stack(item, dim=0)).sum(dim=1)#/item_layer[0].shape[0]
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
        h_u_final = (torch.stack(user, dim=0) * alpha_ul1).mean(dim=0) #/user_layer[0].shape[0
        h_v_final = (torch.stack(item, dim=0) * alpha_vl1).mean(dim=0) #/user_layer[0].shape[0
        # h_u_final = (torch.stack(user, dim=0)).mean(dim=0).cuda()
        # h_v_final = (torch.stack(item, dim=0)).mean(dim=0).cuda()
        # for i in range(10):
        #     print("#"*100)
        if self.layer_number % 2 == 0:
            return learn_user, learn_item, h_u_final, h_v_final, alpha_ul, alpha_vl, Hu, Hv
        return learn_user[0], learn_item[0], h_u_final, h_v_final, alpha_ul, alpha_vl, Hu, Hv
    
    
    def forward2(self, ufea, vfea, UV_adj, VU_adj,adj):
            learn_user = ufea
            learn_item = vfea
            user = []
            item = []
            for i, layer in enumerate(self.encoder):
                if i % 2 == 1:
                    learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
                    user.append(learn_user)
                    item.append(learn_item)
                else:
                    learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
                    user.append(learn_user[0])
                    item.append(learn_item[0])
            self.user = user
            self.item = item
            m1 =nn.Softmax(dim=0)

            alpha_u = []
            alpha_v = []
            # indiviual node attention for each layer
            alpha_u = m1(self.MLP_ul1(torch.stack(user, dim=0)))
            alpha_v = m1(self.MLP_vl1(torch.stack(item, dim=0)))

            # layer embeddings
            Hu = (torch.mul(torch.stack(user, dim=0), alpha_u)).sum(dim=1)
            Hv = (torch.mul(torch.stack(item, dim=0), alpha_v)).sum(dim=1)

            # layer wise attention
            alpha_ul = m1(self.MLP_ul(Hu)).unsqueeze(1)
            alpha_vl = m1(self.MLP_vl(Hv)).unsqueeze(1)

            # final node embeddings
            h_u_final = (torch.mul(torch.stack(user, dim=0), alpha_u)).mean(dim=0)
            h_v_final = (torch.mul(torch.stack(item, dim=0), alpha_v)).mean(dim=0)

            # h_u_final = (torch.stack(user, dim=0)).mean(dim=0).cuda()
            # h_v_final = (torch.stack(item, dim=0)).mean(dim=0).cuda()

            if self.layer_number % 2 == 0:
                return learn_user, learn_item, h_u_final, h_v_final, alpha_ul.squeeze(1), alpha_vl.squeeze(1), Hu, Hv
            return learn_user[0], learn_item[0], h_u_final, h_v_final, alpha_ul.squeeze(1), alpha_vl.squeeze(1), Hu, Hv
    
    

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
        self.user_union = nn.Linear(opt["feature_dim"] *2, opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] *2, opt["feature_dim"])
 
        self.user_union1 = nn.Linear(opt["feature_dim"] *3, opt["feature_dim"])
        self.item_union1 = nn.Linear(opt["feature_dim"] *3, opt["feature_dim"])
 
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
 
            User_h = [F.relu(User_h) , User_n, ufea]
            Item_h = [F.relu(Item_h),Item_n, vfea]
 
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
            User_h = torch.tanh(self.user_union1(User_h))
            Item_h = torch.tanh(self.item_union1(Item_h))
 
            return User_h,Item_h
