import torch
import torch.nn as nn
import torch.nn.functional as F 
import math



class SphereNeuralNetwork(nn.Module):
    def __init__(self):
        super(SphereNeuralNetwork, self).__init__()

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







