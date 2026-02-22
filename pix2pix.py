import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
satellite_dataset = ImageFolder(root="./data/satellite", transform=transform)
map_dataset = ImageFolder(root="./data/map", transform=transform)

# Data loaders
batch_size = 16
satellite_loader = DataLoader(satellite_dataset, batch_size=batch_size, shuffle=True)
map_loader = DataLoader(map_dataset, batch_size=batch_size, shuffle=True)

# Show sample
sample_image, _ = satellite_dataset[0]
plt.imshow(sample_image.permute(1, 2, 0))
plt.title("Sample Image - Satellite Domain")
plt.axis("off")
plt.show()
import torch
import torch.nn as nn

# Generator (U-Net style simplified)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))  # Output scaled to [-1, 1]
        return x

# Discriminator (PatchGAN style simplified)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(64 * 128 * 128, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc(x))
        return x

# Initialize models
G_sat2map = Generator()
D_map = Discriminator()
