import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--indim', type=int, default=784)
    parser.add_argument('--hdim', type=int, default=200)
    parser.add_argument('--zdim', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=51)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wgts_path', type=str, default="weights/best_model_py_project.pth")
    args = parser.parse_args()
    return args