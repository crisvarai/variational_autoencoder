import torch
from torch import nn

class VAE(nn.Module):
    
    def __init__(self, in_dim, hidden_dim=200, z_dim=20):
        super().__init__()
        # encoder init
        self.img_to_hid = nn.Linear(in_dim, hidden_dim)
        self.hidden_to_mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(hidden_dim, z_dim)

        # decoder init
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_img = nn.Linear(hidden_dim, in_dim)

        # activation functions init
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # Encoder Function
    def encode(self, x):
        hidden = self.relu(self.img_to_hid(x))
        mu, sigma = self.hidden_to_mu(hidden), self.hidden_to_sigma(hidden)
        return mu, sigma

    # Decoder Function
    def decode(self, z):
        hidden = self.relu(self.z_to_hidden(z))
        return self.sigmoid(self.hidden_to_img(hidden))

    # Forward
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.rand_like(sigma)
        z_new = mu + sigma*epsilon
        x_rec = self.decode(z_new)          # x reconstructed
        return x_rec, mu, sigma

if __name__ == "__main__":
    BATCH_SIZE = 1
    IMG_SIZE = 28
    IN_DIM = 784
    x = torch.randn(BATCH_SIZE, IMG_SIZE*IMG_SIZE)
    vae_model = VAE(IN_DIM)
    out, mu, sigma = vae_model(x)
    print(f"x_rec: {out.shape} mu: {mu.shape}, sigma: {sigma.shape}")