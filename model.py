import torch
from torch import nn
import torch.nn.functional as F

# . .  the neural network model
class SiameseNetwork(nn.Module):    
    def __init__(self, feature_dim, device='cuda'):
        super(SiameseNetwork, self).__init__()
        
        # . . set the device
        self.device = device 
        
        # . . define the CNN featurizer
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(13*18*64, 128), # 60>29>13,80>39>18
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    # . . the forward propagation method
    def forward(self, img1, img2):
        # . . propagate both images
        feature_one = self.cnn(img1)
        feature_two = self.cnn(img2)
        # . . return the Euclidian distance between two features
        return torch.norm(feature_one - feature_two, dim=-1)

    # . . make a prediction: compute the distance between two images
    def predict(self, x1, x2):
        x1 = torch.from_numpy(x1).float().to(self.device)
        x2 = torch.from_numpy(x2).float().to(self.device)

        with torch.no_grad():
            distance = self.forward(x1,x2).cpu().numpy()

        return distance.flatten()