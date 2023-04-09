"""To train a diffusion model."""

import torch
import logging
from torch.utils.data import DataLoader

from model.VAE import VAE
from train.fit import fit
from utils.load_args import get_args
from train.inference import inference
from utils.data import transform_dataset, load_model

logging.basicConfig(
    filename="runing.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

if __name__ == "__main__":
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = transform_dataset(args.data_path)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=True)
    model = VAE(args.indim, args.hdim, args.zdim)
    
    logging.info("Start training...")
    fit(model=model, 
        train_loader=train_loader, 
        in_dim=args.indim, 
        epochs=args.epochs, 
        lr=args.lr,
        weights_path=args.wgts_path,
        device=DEVICE)
    logging.info("Finished!")

    model = load_model(model, args.wgts_path).to("cpu")

    logging.info("Start inference...")
    for idx in range(10):
        inference(dataset, model, idx, num_examples=5)
    logging.info("Generated Images Saved!")