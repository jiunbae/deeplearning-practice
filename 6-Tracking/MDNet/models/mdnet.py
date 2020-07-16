from collections import OrderedDict

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(inplace=True),
                                    nn.LocalResponseNorm(2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(inplace=True),
                                    nn.LocalResponseNorm(2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True))),
            ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                    nn.ReLU(inplace=True))),
            ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(512, 512),
                                    nn.ReLU(inplace=True)))]))

        # TODO: fill self.branches
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(.5),
                                       nn.Linear(512, 2)) for _ in range(K)])
    
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)

        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Load pre-trained weights
        if model_path.split('.')[-1] == 'pth':
            self.layers.load_state_dict(torch.load(model_path)['shared_layers'])
        elif model_path.split('.')[-1] == 'mat':
            layers = list(scipy.io.loadmat(model_path)['layers'])[0]
            
            for i in range(3):
                weight, bias = layers[i * 4]['weights'].item()[0]
                self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
        
        self.params = OrderedDict()

        def _append(m, n):
            for c in m.children():
                for k, p in c._parameters.items():
                    if p is not None:
                        name = n + ('_bn_' if isinstance(c, nn.BatchNorm2d) else '_') + k
                        if name not in self.params:
                            self.params[name] = p

        for name, module in self.layers.named_children():
            _append(module, name)
        for k, module in enumerate(self.branches):
            _append(module, 'fc6_{:d}'.format(k))

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.reshape(x.size(0), -1)
                if name == out_layer:
                    return x
                
        # TODO: forward x to each branch
        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)
    
    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            p.requires_grad = any([k.startswith(l) for l in layers])

    def optimizer(self, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
        params = OrderedDict()
        
        for k, p in self.params.items():
            if train_all or (not train_all and p.requires_grad):
                params[k] = p

        param_list = []
        for k, p in params.items():
            lr = lr_base
            for l, m in lr_mult.items():
                if k.startswith(l):
                    lr = lr_base * m
            param_list.append({'params': [p], 'lr':lr})
            
        return optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)

    
class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()
