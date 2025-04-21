import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

(trainX, _), (_, _) = mnist.load_data()
trainX = (trainX.astype(np.float32) - 127.5) / 127.5
trainX = trainX.reshape(-1, 28*28)
train_tensor = torch.tensor(trainX)

train_dataset = TensorDataset(train_tensor, torch.zeros(len(train_tensor)))
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.hidden1 = nn.Linear(noise_dim, 1200)
        self.hidden2 = nn.Linear(1200, 1200)
        self.output = nn.Linear(1200, 784)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.hidden1(z))
        x = self.relu(self.hidden2(x))
        x = self.tanh(self.output(x))
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden1 = nn.Linear(784, 240)
        self.hidden2 = nn.Linear(240, 240)
        self.output = nn.Linear(240, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.hidden1(x))
        x = self.leaky_relu(self.hidden2(x))
        x = self.sigmoid(self.output(x))
        return x


def training(real_label=1.0, fake_label=0.0, num_epochs=50, batch_size=100):
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)

            # discriminator training
            optimD.zero_grad()
            real_images_flat = real_images.view(real_images.size(0), -1)
            output_real = D(real_images_flat)
            labels_real = torch.full((real_images.size(0), 1), real_label, device=device)
            loss_real = criterion(output_real, labels_real)

            noise = torch.randn(real_images.size(0), 100, device=device)
            fake_images = G(noise).detach()
            output_fake = D(fake_images)
            labels_fake = torch.full((real_images.size(0), 1), fake_label, device=device)
            loss_fake = criterion(output_fake, labels_fake)

            lossD = (loss_real + loss_fake) / 2
            lossD.backward()
            optimD.step()

            # generator training
            optimG.zero_grad()
            noise = torch.randn(real_images.size(0), 100, device=device)
            fake_images = G(noise)
            output_fake_for_G = D(fake_images)
            labels_for_G = torch.full((real_images.size(0), 1), real_label, device=device)
            lossG = criterion(output_fake_for_G, labels_for_G)
            lossG.backward()
            optimG.step()

        print(f"Epoch {epoch}: D_loss={lossD.item():.4f}, G_loss={lossG.item():.4f}")

        if epoch % 5 == 0:
            z = torch.randn(16, 100, device=device)
            fake_images = G(z)
            fake_images = fake_images.view(-1, 28, 28).cpu().detach()

            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                ax.imshow(fake_images[i], cmap='gray')
                ax.axis('off')
            plt.suptitle(f"Generated Images after Epoch {epoch}", fontsize=14)
            plt.show()



if __name__ == "__main__":
    device = torch.device("cpu")
    G = Generator(noise_dim=100).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    training()