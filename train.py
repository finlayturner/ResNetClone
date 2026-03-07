import torch
import torch.nn as nn
import time

def train(model, train_loader, test_loader, epochs=10, device="cpu"):

    model.to(device)

    # Adam with a small weight decay for L2 regularisation
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # decrease lr after a few epochs if validation loss plateaus
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=0.1)

    # For use in graph plotting
    train_accuracies = []
    train_losses = []
    test_accuracies = []

    # For use in percentage calculator
    total_len = len(train_loader)

    for epoch in range(epochs):

        print(f"----------Epoch {epoch + 1}----------")
        start_time = time.time()

        model.train()

        epoch_loss = 0
        batches = 0

        batch_count = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            loss = criterion(outputs, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            batches += 1

            print(f"\rProgress: {round((batch_count / total_len) * 100, 1)}%", end="", flush=True)

            batch_count += 1
        
        time_taken = round(time.time() - start_time, 2)
        
        avg_loss = epoch_loss / batches
        train_losses.append(avg_loss)

        train_accuracy = (train_correct * 100 / train_total)
        train_accuracies.append(train_accuracy)

        accuracy = evaluate(model, test_loader, device)
        test_accuracies.append(accuracy)

        # step scheduler once per epoch
        scheduler.step()

        # simple early stopping: if test accuracy converges, break out
        if epoch > 2 and test_accuracies[-1] <= max(test_accuracies[:-1]):
            # no improvement in this epoch
            stagnation = sum(1 for acc in test_accuracies[-3:] if acc <= max(test_accuracies[:-1] or [0]))
            if stagnation >= 3:
                print("No improvement on test set for 3 epochs, stopping early.")
                break

        print(f"\rTime taken: {time_taken} seconds")
        print("Loss:", loss.item())
        print(f"Training accuracy: {(train_accuracy):.2f}%")
        print(f"Test accuracy: {accuracy:.2f}%")
    
    return train_losses, train_accuracies, test_accuracies


def evaluate(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total