import torch.nn.functional as F
import torch.nn as nn

class X(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(X, self).__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.convblock(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ResBlock, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,x):
        out = self.residual_conv(x)
        out = self.residual_conv(out)
        out = out + x

        return out

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        #Layer 1
        self.layer1 = nn.Sequential(
            X(in_channels=64, out_channels=128),
            ResBlock(in_channels=128, out_channels=128),
        )

        #Layer 2
        self.layer2 = nn.Sequential(
            X(in_channels=128, out_channels=256)
        )

        #Layer 3
        self.layer3 = nn.Sequential(
            X(256,512),
            ResBlock(512,512)
        )

        self.maxpool = nn.MaxPool2d(4,4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512,10)

    def forward(self,x):
        x = self.prep_layer(x) #32
        x = self.layer1(x)  #16
        x = self.layer2(x) #8
        x = self.layer3(x) #4
        x = self.maxpool(x) #1
        x = self.flatten(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)

