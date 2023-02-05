import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument("--load", type=str, default=None, help="Model name to be loaded (default: create new model)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' for gpu, 'cpu' for cpu) (default: cuda)")

    parser.add_argument("--starting-task", type=int, default=0, help="Index of first task (default: 0)")

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--task-size", type=int, default=2, help="Number of classes per task (default: 2)")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks (default: 5)")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--train-batch-size", type=int, default=128, help="Batch size during training (default: 128)")
    parser.add_argument("--val-batch-size", type=int, default=512, help="Batch size during validation (default: 512)")

    parser.add_argument("--num-sums", type=int, default=5, help="Number of sum nodes (default: 5)")
    parser.add_argument("--num-leaves", type=int, default=5, help="Number of leave nodes (default: 5)")
    parser.add_argument("--num-repetitions", type=int, default=1, help="Number of repetitions of the network (default: 1)")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the network (default: 3)")

    parser.add_argument("--seed", type=int, default=0, help="Seed used for random generator (default: 0)")

    return parser.parse_args()


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


def evaluate(losses, accuracies, epochs=None, tasks=None):
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
    plotLoss(losses, reached_loss, convergence_border, epochs, tasks)

    # Accuracies
    classification_border = None
    for i, acc in enumerate(accuracies):
        if acc > threshold:
            classification_border = i
            break
    plotAccuracy(accuracies, threshold, classification_border, epochs, tasks)

    return convergence_border, classification_border, accuracies[0], accuracies[-1], reached_loss


def plotLoss(losses, min_loss, convergence_border, epochs=None, tasks=None):
    plt.plot(losses)
    # if convergence_border != len(losses) - 1:
    #     plt.axhline(y=min_loss, c='lightgrey')
    #     plt.axvline(x=convergence_border, c='r')
    # plt.title('loss: ' + str(round(losses[-1], 3)))

    if epochs is not None:
        epoch_size = (len(losses) // tasks) // epochs
        task_size = len(losses) // tasks
        for i in range(1, tasks):
            # for j in range(0, epochs, (epochs // 4)):
            #     plt.axvline(x=i * task_size + j * epoch_size, c='lightgrey')
            plt.axvline(x=i * task_size, c='black')

        # ticks = range(1, len(losses), epoch_size * (epochs // 4))
        # labels = list(range(0, epochs, (epochs // 4))) * tasks + [epochs]
        # plt.xticks(ticks, labels)

    plt.show()


def plotAccuracy(accuracies, threshold, classification_border, epochs=None, tasks=None):
    plt.plot(accuracies)
    # if classification_border is not None:
    #     plt.axhline(y=threshold, c='lightgrey')
    #     plt.axvline(x=classification_border, c='r')
    # plt.title('acc: ' + str(round(accuracies[-1], 2)))
    # plt.ylim([-0.1, 1.1])

    if epochs is not None:
        epoch_size = (len(accuracies) // tasks) // epochs
        task_size = len(accuracies) // tasks
        for i in range(1, tasks):
            # for j in range(0, epochs, (epochs // 4)):
            #     plt.axvline(x=i * task_size + j * epoch_size, c='lightgrey')
            plt.axvline(x=i * task_size, c='black')

        # ticks = range(1, len(accuracies), epoch_size * (epochs // 4))
        # labels = list(range(0, epochs, (epochs // 4))) * tasks + [epochs]
        # plt.xticks(ticks, labels)

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
