import matplotlib.pyplot as plt

def plot_metrics(losses, train_accuracies, test_accuracies):

    epochs = range(len(losses))

    plt.figure()

    plt.subplot(1,2,1)
    plt.plot(epochs, losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.title("Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.show()