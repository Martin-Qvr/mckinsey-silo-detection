"""
Contained pretrained EffnetModel
"""
import torchvision
from torch import nn

class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet_model = torchvision.models.efficientnet_b0(pretrained=True)
        
        # for param in effnet_model.features.parameters():
        #     param.requires_grad = False
        self.effnet_model.features[0] = nn.Sequential(
                   nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.SiLU(inplace=True)
                    )
        self.effnet_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), 
            
            nn.Linear(in_features=1280, 
                        out_features=1, # same number of output units as our number of classes
                        bias=True))
        
    def forward(self, x):
        return self.effnet_model(x)
        
