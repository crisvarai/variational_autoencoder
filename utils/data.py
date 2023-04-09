import torch
import torchvision.datasets as datasets
from torchvision import transforms

def transform_dataset(path):
    data_transforms = transforms.Compose([
        transforms.ToTensor(), # Scale between [0, 1] 
    ])
    dataset = datasets.MNIST(root=path, download=True, transform=data_transforms, train=True)                             
    return dataset

def load_model(model, weights_path):
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)
    return model