import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_in,n_out):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)
        self.sigm = nn.Sigmoid()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, node, s_bias=None):
        S = S.expand_as(node) # batch * hidden_dim
        score = torch.squeeze(self.f_k(node, S),1) # batch
        if s_bias is not None:
            score += s_bias

        return self.sigm(score)
#####
#####
class myDGI(nn.Module):
    def __init__(self, opt):
        super(myDGI, self).__init__()
        self.opt = opt
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.lin = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"]).cuda()
        self.lin_sub = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"]).cuda()
        self.disc = Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda()
        for m in self.modules():
            self.weights_init(m)
            
        self.MLP_u = nn.Linear(opt["hidden_dim"], 1).cuda()#opt["hidden_dim"])
        self.MLP_u1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        self.MLP_v = nn.Linear(opt["hidden_dim"], 1).cuda()#opt["hidden_dim"])
        self.MLP_v1 = nn.Linear(opt["hidden_dim"], 1).cuda()

        self.MLP_ul = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()
        self.MLP_ul1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        self.MLP_vl = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()
        self.MLP_vl1 = nn.Linear(opt["hidden_dim"], 1).cuda()

        self.MLP_uvl = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()
        self.MLP_uvl1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        self.MLP_cuvl = nn.Linear(opt["hidden_dim"], opt["hidden_dim"]).cuda()
        self.MLP_cuvl1 = nn.Linear(opt["hidden_dim"], 1).cuda()
        
        self.ulist = []
        self.ilist = []
        for i in range(self.opt["GNN"]):
            self.ulist.append(Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda())
            self.ilist.append(Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda())
    
        self.discu1 = Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda()
        self.discv1 = Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda()
        self.discu2 = Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda()
        self.discv2 = Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda()

        self.disc = Discriminator(opt["hidden_dim"],opt["hidden_dim"]).cuda()
        
        self.mm = nn.Linear(opt["hidden_dim"]*opt["GNN"], opt["hidden_dim"]).cuda()
        self.mm1 = nn.Linear(opt["hidden_dim"]*opt["GNN"], opt["hidden_dim"]).cuda()
        
        self.mm2 = nn.Linear(opt["hidden_dim"]*2, opt["hidden_dim"]).cuda()
        self.mm3 = nn.Linear(opt["hidden_dim"]*opt["GNN"], opt["hidden_dim"]).cuda()

        self.mm4 = nn.Linear(opt["hidden_dim"]*opt["GNN"], opt["hidden_dim"]).cuda()


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    #
    def forward(self, user_layer, item_layer, corr_user_layer, corr_item_layer, UV, VU, CUV, CVU, user_One, item_One,neg_item_One, alpha_ul,  alpha_vl, Hu, Hv, alpha_cul, alpha_cvl, Huc, Hvc , h_u_final ,h_v_final, h_negu_final, h_negv_final, msk=None, samp_bias1=None, samp_bias2=None):
    
        user_layer_l = []
        item_layer_l = []
        item_neg = []
        for i in range(self.opt["GNN"]):
            user_layer_l.append(torch.index_select(user_layer[i], 0, user_One).cuda())
            item_layer_l.append(torch.index_select(item_layer[i], 0, item_One).cuda())
            item_neg.append(torch.index_select(item_layer[i], 0, neg_item_One).cuda())
            
        m1 =nn.Softmax(dim=0)
        

        HU =  (Hu * alpha_ul).sum(dim=0)
        
        HV =  (Hv * alpha_vl).sum(dim=0)
        
        HUC =  (Huc * alpha_cul).sum(dim=0)
        
        HVC =  (Hvc * alpha_cvl).sum(dim=0)
        
        H_uv = F.relu(self.mm2(torch.cat((alpha_ul * Hu, alpha_vl * Hv), dim=1)))
        H_uvc = F.relu(self.mm2(torch.cat((alpha_cul*Huc , alpha_cvl * Hvc), dim=1)))
        
        alpha_uvl = m1(self.MLP_ul(nn.ReLU()(self.MLP_uvl(H_uv))))
        alpha_cuvl = m1(self.MLP_ul(nn.ReLU()(self.MLP_uvl(H_uvc))))
 
        #HCM - overall network      
        H_UV = (H_uv * alpha_uvl).sum(dim=0)
        H_UVC = (H_uvc * alpha_cuvl).sum(dim=0)
                       
        
        
        ########################### LOSS CALCULATION ###############################
        # Global vectors (for user and item)
        dgi_loss = 0
        
        for i in range(0,self.opt["GNN"]):


            pos_u = self.ulist[i]((Hu[i]),(user_layer_l[i]))
            pos_v = self.ilist[i]((Hv[i]), (item_layer_l[i]))
 
            neg_u = self.ulist[i]((Huc[i]),(user_layer_l[i]))
            neg_v = self.ilist[i]((Hvc[i]),(item_layer_l[i]))

            prob = torch.cat((pos_u,neg_u))
            label = torch.cat((torch.ones_like(pos_u), torch.zeros_like(neg_u)))

            probv = torch.cat((pos_v,neg_v))
            labelv = torch.cat((torch.ones_like(pos_v), torch.zeros_like(neg_v)))

            dgi_loss += self.criterion(prob, label) + self.criterion(probv,labelv)        


            pos_U = self.discu1((HU), (user_layer_l[i])) 
            neg_U = self.discu1((HUC), (user_layer_l[i])) 

            prob = torch.cat((pos_U, neg_U))
            label = torch.cat((torch.ones_like(pos_U), torch.zeros_like(neg_U)))

            pos_V = self.discv1((HV), (item_layer_l[i]))
            neg_V = self.discv1((HVC), (item_layer_l[i]))
            
            probv = torch.cat((pos_V, neg_V))
            labelv = torch.cat((torch.ones_like(pos_V), torch.zeros_like(neg_V)))

            dgi_loss += ((self.criterion(prob, label) + self.criterion(probv,labelv)))#/251



            pos_UV = self.discu2((H_UV), (user_layer_l[i]))
            neg_UV = self.discu2((H_UVC), (user_layer_l[i]))

            prob = torch.cat((pos_UV, neg_UV))
            label = torch.cat((torch.ones_like(pos_UV), torch.zeros_like(neg_UV)))

            pos_VU = self.discv2((H_UV), (item_layer_l[i]))
            neg_VU = self.discv2((H_UVC), (item_layer_l[i]))

            probv = torch.cat((pos_VU, neg_VU))
            labelv = torch.cat((torch.ones_like(pos_VU), torch.zeros_like(neg_VU)))

            dgi_loss +=  (self.criterion(prob, label) + self.criterion(probv,labelv))#/251
        

        return dgi_loss
