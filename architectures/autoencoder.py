import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_size=115, hidden_size=4):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, hidden_size),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_size),
            nn.LeakyReLU(0.2),
        )

    def dimension_reduction(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
