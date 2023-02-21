import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
sys.path.append('/Users/alejandrobonell/pytorch_guide/2d_sphere_nn')
from src.model import SphereNeuralNetwork
import os
from os.path import join, dirname
import logging
from dotenv import load_dotenv


    
def train_one_epoch(model, train_loader, optimizer, loss_fn):

    running_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(inputs).squeeze()
        target = labels.to(torch.float32)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss


def evaluate(model, dataloader):
    acc = 0
    for input, target in dataloader:
        output = model(input)
        _,pred = torch.max(output, 1)
        _,target = torch.max(target, 1)
        acc += torch.sum(target==pred).detach().numpy()
    return acc


def predict(model, inputs):
    preds = model.predict(inputs).detach().numpy()
    return int(preds)



def predict_pipeline(model_, inputs):
    return predict(model_, inputs)


