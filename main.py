import torch

from models.resnet import ResNet18
from data import get_dataloaders
from train import train
from plot import plot_metrics

def main():

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device set to {device}")
    
    print(f"Loading data...")
    train_loader, test_loader = get_dataloaders()
    print(f"Data loaded")

    print(f"Initialising model")
    model = ResNet18()
    print(f"Model initialised")

    print(f"Beginning training...")
    train_losses, train_accuracies, test_accuracies = train(model, train_loader, test_loader, epochs=10, device=device)
    print(f"Training completed")

    plot_metrics(train_losses, train_accuracies, test_accuracies)
    
if __name__ == "__main__":
    main()