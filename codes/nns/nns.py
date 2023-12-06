import torch
from torch import (
    nn,
    functional as F,
)


def make_model(model_config: dict):
    return


class Model(nn.Module):
    def __init__(self):
        pass


class Elephant(nn.Module):
    def __init__(self, sigma=2.0, d=20.0):
        super().__init__()
        self.sigma = sigma
        self.d = d
    
    def forward(self, x):
        return 1.0/(1.0+torch.pow(torch.abs(x/self.sigma), self.d))


class MLP(nn.Module):
    def __init__(self, dims, last_layer_bias=None):
        super().__init__()
        self.model = self.make_model(dims, last_layer_bias)
    
    def make_model(self, dims, last_layer_bias):
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if last_layer_bias is not None:
                with torch.no_grad():
                    layers[-1].bias.fill_(last_layer_bias)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class EMLP(nn.Module):
    def __init__(self, dims, last_layer_bias="elephant"):
        super().__init__()
        self.model = self.make_model(dims, last_layer_bias)
    
    def make_model(self, dims, last_layer_bias):
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            with torch.no_grad():
                layers[-1].bias.uniform_(-1.0, 1.0)
            if i == len(dims) - 2:
                layers.append(Elephant())
            else:
                layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        with torch.no_grad():
            if last_layer_bias == "elephant":
                layers[-1].bias.uniform_(-1.0, 1.0)
            elif last_layer_bias is not None:
                layers[-1].bias.fill_(last_layer_bias)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
        

            
