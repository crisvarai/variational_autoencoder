import torchvision.datasets as datasets
from torchvision import transforms

def transform_dataset(path):
    data_transforms = transforms.Compose([
        transforms.ToTensor(), # Scale between [0, 1] 
    ])
    dataset = datasets.MNIST(root=path, download=True, transform=data_transforms, train=True)                             
    return dataset