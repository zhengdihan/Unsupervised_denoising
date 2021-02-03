import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=3, num_channels=[64, 128], need_sigmoid=False, res_out=True):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential()
        
        for i in range(len(num_channels)):
            if i == 0:
                self.model.add_module(str(len(self.model)+1), nn.Conv2d(ch_in, num_channels[i], kernel_size=3, stride=1, padding=1))
            else:
                self.model.add_module(str(len(self.model)+1), nn.Conv2d(num_channels[i-1], num_channels[i], kernel_size=3, stride=1, padding=1))
                
            self.model.add_module(str(len(self.model)+1), nn.BatchNorm2d(num_channels[i]))
            self.model.add_module(str(len(self.model)+1), nn.ReLU(inplace=True))
        
        self.model.add_module(str(len(self.model)+1), nn.Conv2d(num_channels[-1], ch_out, kernel_size=3, stride=1, padding=1))
        if need_sigmoid:
            self.model.add_module(str(len(self.model)+1), nn.Sigmoid())
            
        self.res_out = res_out

    def forward(self, x):
        if self.res_out:
            return x - self.model(x)
        else:
            return self.model(x)
    
