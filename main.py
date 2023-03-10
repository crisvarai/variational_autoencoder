"""To train a diffusion model."""

import torch
from torch.utils.data import DataLoader

from model.VAE import VAE
from train.fit import fit
from utils.load_args import get_args
from utils.data import transform_dataset

if __name__ == "__main__":
    args = get_args()
    DATASET_PATH = args.data_path

    IN_DIM = args.indim
    H_DIM = args.hdim
    Z_DIM = args.zdim

    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    LR = args.lr

    WEIGHTS_PATH = args.wgts_path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = transform_dataset(DATASET_PATH)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = VAE(IN_DIM, H_DIM, Z_DIM)
    
    fit(model=model, 
        train_loader=train_loader, 
        in_dim=IN_DIM, 
        epochs=EPOCHS, 
        lr=LR,
        weights_path=WEIGHTS_PATH,
        device=DEVICE)