import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt


def get_datasets(task_size, task, train_batch_size, test_batch_size):
    from_class = task * task_size
    to_class = from_class + task_size

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_idx = ((train_dataset.targets >= from_class) & (train_dataset.targets < to_class))
    test_idx = ((test_dataset.targets >= from_class) & (test_dataset.targets < to_class))

    train_dataset.data = train_dataset.data[train_idx]
    test_dataset.data = test_dataset.data[test_idx]

    train_dataset.targets = train_dataset.targets[train_idx] - from_class
    test_dataset.targets = test_dataset.targets[test_idx] - from_class

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def test(model, data, labels):
    correct_predictions, total_predictions = 0, 0
    with torch.no_grad():
        outputs = model(data)
        _, predictions = torch.max(outputs, dim=1)
        total_predictions += labels.size(0)
        correct_predictions += (predictions == labels).sum().item()
    return correct_predictions / total_predictions


def evaluate(losses, accuracies, epochs=None):
    epsilon = 0.15
    threshold = 0.95

    # Losses
    min_loss = losses[0]
    convergence_border = 0
    for i, loss in enumerate(losses):
        if loss < min_loss * (1 + epsilon):
            min_loss = loss
            convergence_border = i
    reached_loss = sum(losses[convergence_border:]) / (len(losses) - convergence_border)
    plotLoss(losses, reached_loss, convergence_border, epochs)

    # Accuracies
    classification_border = None
    for i, acc in enumerate(accuracies):
        if acc > threshold:
            classification_border = i
            break
    plotAccuracy(accuracies, threshold, classification_border, epochs)

    return convergence_border, classification_border, accuracies[0], accuracies[-1], reached_loss


def plotLoss(losses, min_loss, convergence_border, epochs=None):
    plt.plot(losses)
    if convergence_border != len(losses) - 1:
        plt.axhline(y=min_loss, c='lightgrey')
        plt.axvline(x=convergence_border, c='r')
    plt.title('loss: ' + str(round(losses[-1], 3)))

    if epochs is not None:
        epoch = len(losses) // epochs
        for i in range(1, 1 + epochs):
            plt.axvline(x=epoch * i, c='lightgrey')

    plt.show()


def plotAccuracy(accuracies, threshold, classification_border, epochs=None):
    plt.plot(accuracies)
    if classification_border is not None:
        plt.axhline(y=threshold, c='lightgrey')
        plt.axvline(x=classification_border, c='r')
    plt.title('acc: ' + str(round(accuracies[-1], 2)))
    plt.ylim([-0.1, 1.1])

    epoch = len(accuracies) // epochs
    for i in range(1, 1 + epochs):
        plt.axvline(x=epoch * i, c='lightgrey')

    plt.show()


def printProgress(time, acc, loss, batch, batches, epoch, epochs, task=None, tasks=None):
    if task is None:
        s = int(time * (epochs * batches - (epoch * batches + batch)))
        prog = round((epoch * batches + batch) / (epochs * batches) * 100, 2)
    else:
        s = int(time * (tasks * epochs * batches - (task * epochs * batches + epoch * batches + batch)))
        prog = round((task * epochs * batches + epoch * batches + batch) / (tasks * epochs * batches) * 100, 2)
    h = s // 3600
    s = s % 3600
    m = s // 60
    s = s % 60
    itps = round(1 / time, 3)
    acc = round(acc * 100, 2)
    if task is None:
        print("\rEpoch: {} / {} - Batch: {} / {} ( Progress: {}% ) | Time Remaining: {}h :{}m :{}s ({} it/s) | Accuracy: {}% | Loss: {}"
                .format(epoch + 1, epochs, batch + 1, batches, prog, h, m, s, itps, acc, loss), end="")
    else:
        print("\rTask: {} / {} - Epoch: {} / {} - Batch: {} / {} ( Progress: {}% ) | Time Remaining: {}h :{}m :{}s ({} it/s) | Accuracy: {}% | Loss: {}"
            .format(task + 1, tasks, epoch + 1, epochs, batch + 1, batches, prog, h, m, s, itps, acc, loss), end="")
