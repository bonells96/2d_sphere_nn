import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math


def generate_2d_uniform_vectors(n) -> torch.Tensor:
    return 2*torch.rand(n,2) - 1

def is_in_2d_sphere(x:torch.Tensor, radius:float, center=torch.Tensor([0,0])) -> torch.Tensor:
    y = torch.sum((x - center)**2, 1)
    y[y<radius**2] = 0
    y[y>radius**2] = 1
    return torch.nn.functional.one_hot(y.long(), num_classes=2)

class GenerateUnitSphere2dDataset(Dataset):
    def __init__(self, n: int, radius: float):
        self.n = n
        self.coordinates = generate_2d_uniform_vectors(self.n)
        self.labels = is_in_2d_sphere(self.coordinates, radius)

    def __len__(self):
        return len(self.labels)
		
    def __getitem__(self, idx):
	    return self.coordinates[idx], self.labels[idx]

    
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
    return model.predict(inputs)

