import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        
        self.features = vgg16.features
        
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        
        # weight copy
        self.fconn[0].weight.data = vgg16.classifier[0].weight.view(self.fconn[0].weight.data.size())
        self.fconn[3].weight.data = vgg16.classifier[3].weight.view(self.fconn[3].weight.data.size())
        
        # bias copy
        self.fconn[0].bias.data = vgg16.classifier[0].bias.view(self.fconn[0].bias.data.size())
        self.fconn[3].bias.data = vgg16.classifier[3].bias.view(self.fconn[3].bias.data.size())
        
        # new score layer
        self.score = nn.Conv2d(4096, num_classes, 1)
        
    def forward(self, x):
        features = self.features(x)
        fconn = self.fconn(features)
        score = self.score(fconn)
        
        return F.upsample(score, scale_factor=32, mode='bilinear', align_corners=True)
