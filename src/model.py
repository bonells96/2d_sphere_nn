import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

class ShallowNetwork(nn.Module):
    def __init__(self):
        super(ShallowNetwork, self).__init__()

        self.h1 = nn.Linear(2, 128)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):

        out = F.relu(self.h1(x))

        out = F.sigmoid(self.fc(out))

        return out
    
    def predict(self, x):
        
        out = self.forward(x)
        _,pred = torch.max(out, 1)
        return pred 



class DeepNetwork(nn.Module):
    def __init__(self):
        super(DeepNetwork, self).__init__()

        self.h1 = nn.Linear(2, 4)
        self.h2 = nn.Linear(4, 8)
        self.h3 = nn.Linear(8, 16)
        self.h4 = nn.Linear(16, 32)
        self.h5 = nn.Linear(32, 64)
        self.h6 = nn.Linear(64, 128)

        self.fc = nn.Linear(128, 2)

    def forward(self, x):

        out = self.h1(x)
        out = self.h2(out)
        out = self.h3(out)
        out = self.h4(out)
        out = self.h5(out)
        out = self.h6(out)

        out = F.sigmoid(self.fc(out))

        return out
    




