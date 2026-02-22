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