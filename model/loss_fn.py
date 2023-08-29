import torch
import torch.nn as nn
import random
import numpy as np
import math
import torch.nn.functional as F

class NLP_Projector(nn.Module):
    def __init__(self, proj_type, inp_dim, hid_dim, out_dim, device, temperature=None):
        super(NLP_Projector, self).__init__()
        self.layer = make_projector(proj_type, inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, device=device, temperature=temperature)
    def forward(self, x):
        return self.layer(x)
    
class Vision_Projector(nn.Module):
    def __init__(self, proj_type, inp_dim, hid_dim, out_dim, device, temperature=None, shape=None):
        super(Vision_Projector, self).__init__()
        self.device = device
        self.proj_type = proj_type
        self.avg_pool_f = False
        self.avg_pool = None
        self.avg_base = 2

        if "avg" in self.proj_type:
            if shape <= self.avg_base:
                self.avg_pool_f == False
            else:
                print("[CL Loss] use nn.AdaptiveAvgPool2d({},{})".format(self.avg_base,self.avg_base))
                reshape = int(shape/self.avg_base)
                inp_dim = int(inp_dim / (shape*shape)) * reshape * reshape
                self.avg_pool = nn.AdaptiveAvgPool2d((reshape, reshape))
                self.avg_pool_f = True

        self.layer = nn.Sequential(Flatten(), make_projector(proj_type, inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, device=device, temperature=temperature))

    def forward(self, x):
        if self.avg_pool_f == True:
            # print("before,",x.shape)
            x = self.avg_pool(x)
            # print("End,", x.shape)
        output = self.layer(x)
        return output

class LocalLoss(nn.Module):
    def __init__(self, temperature=0.1, input_dim = 128, hid_dim = 512, out_dim = 1024, 
                 num_classes=None, proj_type=None, device=None):
        super(LocalLoss, self).__init__()
        self.temperature = temperature
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        if device != None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if "cpu" in self.device:
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        self.proj_type = proj_type.replace(' ', '').split(",") if proj_type != None else None
        self.projector = None

        if self.proj_type != None:
            self.proj_type, self.proj_dim = self._parser_type(self.proj_type, hid_dim, out_dim, mode="proj")

        # if "dcl" in proj_type:
        #     self.loss_type = 'dcl'
        #     print("[CL Loss] Use dcl loss")
        # else:
        #     self.loss_type = 'infoNCE'
        #     print("[CL Loss] Use infoNCE loss")
    
    def _parser_type(self, _type, hid_dim=None, out_dim=None, mode="proj"):
        if _type==None: return None, [None, None] 

        _set = []
        _dim = []
        for item in _type:
            if item.isdigit():
                _dim.append(int(item))
            else:
                _set.append(item)

        if mode == "proj":
            if "i" in _set:
                assert len(_dim) == 0, "[CL Loss Error] You can not set dimension of identity function in projection head."
                _dim = [None, None] 
            elif len(_dim) == 0:
                _dim = [out_dim, hid_dim] 
            elif "l" in _set:
                assert len(_dim) == 1, "[CL Loss Error] You only can set one dimension (output dimension) of linear function in projection head."
                _dim = [_dim[-1], hid_dim] 
            elif len(_dim) == 1:
                _dim = [out_dim, _dim[-1]]
            else:
                assert len(_dim) == 2, "[CL Loss Error] You only can set two dimension (hidden and output dimension) in projection head."
                _dim = [_dim[-1], _dim[-2]]
                
        return _set, _dim

    def _contrastive_loss(self, x, label):
        x = self.projector(x)
        x =  nn.functional.normalize(x)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).type(self.tensor_type)
        denom_mask = torch.scatter(torch.ones_like(mask, device=x.device), 1, torch.arange(batch_size, device=x.device).view(-1, 1), 0)
        logits = torch.div(torch.matmul(x, x.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        denom = torch.exp(logits) * denom_mask
        # if self.loss_type == 'dcl':
        #     invert_make = torch.ones_like(mask, device=x.device) - mask
        #     denom = torch.exp(logits) * invert_make #denom_mask
        # else:
        #     denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()
        return loss


    def training_mode(self, x, label):
        loss_all = 0
        hat_y = None
        if self.projector != None:
            loss_all += self._contrastive_loss(x, label)
        return loss_all


    def forward(self, x, label=None):
        loss = self.training_mode(x, label)
        return loss

class VisionLocalLoss(LocalLoss):
    def __init__(self, temperature=0.1, input_dim=128, hid_dim=512, out_dim=1024, c_in = 256, shape = 32, 
                 num_classes=None, proj_type=None, device=None):

        super(VisionLocalLoss, self).__init__(temperature, input_dim, hid_dim, out_dim , num_classes, proj_type, device)

        self.input_dim = int(c_in * shape * shape)
        if self.proj_type != None:
            self.projector = Vision_Projector(self.proj_type, inp_dim=self.input_dim, out_dim=self.proj_dim[0], hid_dim=self.proj_dim[1], device=self.device, temperature=self.temperature, shape=shape)

class NLPLocalLoss(LocalLoss):
    def __init__(self, temperature=0.1, input_dim=300, hid_dim=300, out_dim=300,
                 num_classes=None, proj_type=None, device=None):

        super(NLPLocalLoss, self).__init__(temperature, input_dim, hid_dim, out_dim , num_classes, proj_type, device)
        if self.proj_type != None:
            self.projector = NLP_Projector(self.proj_type, inp_dim=self.input_dim, out_dim=self.proj_dim[0], hid_dim=self.proj_dim[1], device=self.device, temperature=self.temperature)

class NLP_Tail(nn.Module):
    def __init__(self, inp_dim, out_dim, hid_dim=100, act_fun = nn.Tanh(), device=None):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(inp_dim, hid_dim), act_fun, nn.Linear(hid_dim, out_dim))
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x):
        return self.layer(x)
    
class vision_Tail(nn.Module):
    def __init__(self, num_classes, in_channels, shape,  hid_dim=500, device=None):
        super().__init__()
        self.device = device
        self.layer = nn.Sequential(Flatten(),
                                #    nn.Linear(in_channels, hid_dim, bias=False),
                                #    nn.BatchNorm1d(hid_dim),
                                #    nn.ReLU(inplace=True), # hidden layer
                                #    nn.Linear(hid_dim, num_classes)) # output layer
                                nn.Linear(in_channels, hid_dim), nn.Sigmoid(), nn.Linear(hid_dim, num_classes))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        output = self.layer(x)
        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, input_neurons = 128, mid_neurons = 512, out_neurons = 1024, 
                 c_in = 256, shape = 32, device=None, num_classes=None, proj_type='m'):
        super(ContrastiveLoss, self).__init__()
        
        self.temperature = temperature

        if device != None:
            self.device = device
        else:
            # self.device = "cpu"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if "cpu" in self.device:
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        print(c_in, shape, c_in * shape * shape)
        
        input_neurons = int(c_in * shape * shape)

        self.proj_type = proj_type.split(",")
        self.projector = nn.Sequential(Flatten(), make_projector(proj_type, input_neurons, mid_neurons, out_neurons, self.device))

        if "predict" in self.proj_type:
            self.classifier = vision_Tail(num_classes=num_classes, in_channels=input_neurons, shape=shape, device=self.device)
            print("[CL Loss] Use local classifier, in_dim: {}, out_dim: {}, Device: {}".format(input_neurons, num_classes, self.device))
    
    def train_mode(self, x, label):
        label2 = label.clone()
        x1 = self.projector(x)
        x1 =  nn.functional.normalize(x1)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).type(self.tensor_type)
        denom_mask = torch.scatter(torch.ones_like(mask, device=x.device), 1, torch.arange(batch_size, device=x.device).view(-1, 1), 0)
        logits = torch.div(torch.matmul(x1, x1.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()

        if "predict" in self.proj_type:
            y = self.classifier(x.detach() if "detach" in self.proj_type else x)
            label_loss = self.classifier.loss(y, label2)
            loss = loss + label_loss

        return loss
    
    def inference(self, x, label):
        if "predict" in self.proj_type:
            y = self.classifier(x)
            return y
        else:
            return None

    def forward(self, x, label=None):
        if self.training:
            return self.train_mode(x, label)
        else:
            return self.inference(x, label)

class PredSimLoss(nn.Module):
    def __init__(self, temperature = 0.1, input_neurons = 2048, c_in = 256, shape = 32):
        super().__init__()
        num_classes = 200
        self.conv_loss = nn.Conv2d(c_in, c_in, 3, stride=1, padding=1, bias=False)
        self.decoder_y = nn.Linear(input_neurons, num_classes)
        # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
        ks_h, ks_w = 1, 1
        dim_out_h, dim_out_w = shape, shape
        dim_in_decoder = c_in*dim_out_h*dim_out_w
        while dim_in_decoder > input_neurons and ks_h < shape:
            ks_h*=2
            dim_out_h = math.ceil(shape / ks_h)
            dim_in_decoder = c_in*dim_out_h*dim_out_w
            if dim_in_decoder > input_neurons:
               ks_w*=2
               dim_out_w = math.ceil(shape / ks_w)
               dim_in_decoder = c_in*dim_out_h*dim_out_w 
        if ks_h > 1 or ks_w > 1:
            pad_h = (ks_h * (dim_out_h - shape // ks_h)) // 2
            pad_w = (ks_w * (dim_out_w - shape // ks_w)) // 2
            self.avg_pool = nn.AvgPool2d((ks_h,ks_w), padding=(0, 0))
        else:
            self.avg_pool = None
    def forward(self, h, y):
        y_onehot = nn.functional.one_hot(y, num_classes=200).float()
        h_loss = self.conv_loss(h)
        Rh = similarity_matrix(h_loss)
        
        if self.avg_pool is not None:
            h = self.avg_pool(h)
        y_hat_local = self.decoder_y(h.view(h.size(0), -1))
        
        Ry = similarity_matrix(y_onehot).detach()
        loss_pred = (1-0.99) * F.cross_entropy(y_hat_local,  y.detach())
        loss_sim = 0.99 * F.mse_loss(Rh, Ry)
        loss = loss_pred + loss_sim
        
        return loss

def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R

class NLPContrastiveLoss(nn.Module):
    def __init__(self,  inp_dim, out_dim, hid_dim=100, temperature=0.1, proj_type="i", class_num=None, device=None):
        super(NLPContrastiveLoss, self).__init__()
        self.temperature = temperature
        
        if device != None:
            self.device = device
        else:
            # self.device = "cpu"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if "cpu" in self.device:
            self.tensor_type = torch.FloatTensor
        else:
            self.tensor_type = torch.cuda.FloatTensor

        self.proj_type = proj_type.split(",")
        self.projector = make_projector(self.proj_type, inp_dim, out_dim, hid_dim, self.device)

        if "predict" in self.proj_type:
            self.classifier = NLP_Tail(inp_dim, class_num, hid_dim, act_fun = nn.Tanh(), device=self.device)
            print("[CL Loss] Use local classifier, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, class_num, hid_dim, device)
                + ", detach: " + "use" if "detach" in self.proj_type else "non")

    def train_mode(self, x, label):
        label2 = label.clone()
        x1 = self.projector(x)
        x1 =  nn.functional.normalize(x1)
        label = label.view(-1, 1)
        batch_size = label.shape[0]
        mask = torch.eq(label, label.T).type(self.tensor_type)
        denom_mask = torch.scatter(torch.ones_like(mask, device=x.device), 1, torch.arange(batch_size, device=x.device).view(-1, 1), 0)
        logits = torch.div(torch.matmul(x1, x1.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        denom = torch.exp(logits) * denom_mask
        prob = logits - torch.log(denom.sum(1, keepdim=True))
        loss = (denom_mask * mask * prob).sum(1) / mask.sum(1)
        loss = -loss
        loss = loss.view(1, batch_size).mean()

        if "predict" in self.proj_type:
            y = self.classifier(x.detach() if "detach" in self.proj_type else x)
            label_loss = self.classifier.loss(y, label2)
            loss = loss + label_loss

        return loss
    
    def inference(self, x, label):
        if "predict" in self.proj_type:
            y = self.classifier(x)
            return y
        else:
            return None

    def forward(self, x, label=None):

        if self.training:
            return self.train_mode(x, label)
        else:
            return self.inference(x, label)

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def make_projector(proj_type, inp_dim, hid_dim, out_dim, device, temperature=None):
    if "i" in proj_type:
        # Identity function
        print("[CL Loss] Type: Identity function, Device: {}".format(device))
        return nn.Identity()
    elif "l" in proj_type:
        # Linear
        print("[CL Loss] Type: Linear, in_dim: {}, out_dim: {}, Device: {}".format(inp_dim, out_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(nn.Linear(inp_dim, out_dim))
    elif "m" in proj_type:
        # MLP
        print("[CL Loss] Type: MLP, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, out_dim, hid_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(nn.Linear(inp_dim, hid_dim), 
                             nn.ReLU(), 
                             nn.Linear(hid_dim, out_dim))
    elif 'mb' in proj_type:
        print("[CL Loss] Type: MLP with BN, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, out_dim, hid_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(nn.Linear(inp_dim, hid_dim, bias=False),
                             nn.BatchNorm1d(hid_dim),
                             nn.ReLU(inplace=True), # hidden layer
                             nn.Linear(hid_dim, out_dim)) # output layer
    elif 'SimSiamMLP' in proj_type:
        print("[CL Loss] Type: SimSiamMLP, in_dim: {}, out_dim: {}, h_dim: {}, Device: {}".format(inp_dim, out_dim, hid_dim, device), end="")
        print(", Temperature: {}".format(temperature) if temperature != None else "\n")
        return nn.Sequential(
            nn.Linear(inp_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
            )
    else:
        raise RuntimeError("ContrastiveLoss: Error setting of the projective head")