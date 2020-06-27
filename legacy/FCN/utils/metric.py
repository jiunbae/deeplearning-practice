import numpy as np
import torch


def compute_confusion(pred, gt, nc=21):
    p, g = pred.view(-1), gt.view(-1)

    res = p * nc + g
    res[g>20] = -1

    val, cnt = np.unique(res.numpy(), return_counts=True)

    conf = torch.zeros(nc*nc)
    for i in range(1, val.size):
        conf[int(val[i])] = int(cnt[i])
    conf = conf.view(nc,nc)
    return conf
    
def compute_meanIU(conf):
    r = torch.sum(conf, dim=0)
    c = torch.sum(conf, dim=1)
    d = torch.zeros(conf.size(0))
    
    for i in range(0, conf.size(0)):
        d[i] = conf[i,i]
        
    mean = torch.mean(d / (r+c-d))
    mean[torch.isnan(mean)] = 0
    
    return mean
