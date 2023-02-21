
import torch
from torch.utils.data import Dataset

def generate_2d_uniform_vectors(n) -> torch.Tensor:
    return 2*torch.rand(n,2) - 1

def is_in_2d_sphere(x:torch.Tensor, radius:float, center=torch.Tensor([0,0])) -> torch.Tensor:
    y = torch.sum((x - center)**2, 1)
    y[y<radius**2] = 1
    y[y>radius**2] = 0
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

