import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data

# Load and normalize MNIST
(trainX, _), _ = load_data()
trainX = (np.float32(trainX) - 127.5) / 127.5  # Normalize to [-1, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get a minibatch
def get_minibatch(batch_size):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    return torch.tensor(trainX[indices], dtype=torch.float32).view(batch_size, -1).to(device)

# Sample noise
def sample_noise(batch_size, dim=100):
    return torch.randn(batch_size, dim, device=device)

# Generator network
class Generator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1200, output_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=1200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # Paper uses ReLU; LeakyReLU is also okay though
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
