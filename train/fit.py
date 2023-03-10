import torch
import numpy as np
from torch import nn
from tqdm import tqdm

def fit(model, train_loader, in_dim, epochs, lr, weights_path, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="sum")
    train_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = []
        loop = tqdm(train_loader)
        for step, (x, _) in enumerate(loop):
            # Forward pass
            x = x.to(device).view(x.shape[0], in_dim)
            x_rec, mu, sigma = model(x)

            # Compute loss
            rec_loss = loss_fn(x_rec, x)         # reconstruction loss      
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop
            loss = rec_loss + kl_div
            train_loss.append(loss.item())
            mean_tl = np.mean(train_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Loss {mean_tl}")

            if epoch % 5 == 0 and step == 0 and epoch != 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {mean_tl}")
                torch.save(model.state_dict(), weights_path)
                print("WEIGHTS-ARE-SAVED")
        train_losses.append(mean_tl)
    return train_losses