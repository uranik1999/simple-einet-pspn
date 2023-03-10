import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from matplotlib import pyplot as plt

import argparse


def analyseWeights(model, columns=None):
    pspn_weights = []
    for current_column in model.columns:
        pspn_weights.append([])
        for i, layer in enumerate(current_column.layers):
            layer_weights = F.softmax(layer.weights, dim=1)
            pspn_weights[-1].append([])
            for j in range(layer.column_index + 1):
                pspn_weights[-1][i].append(
                    torch.sum(layer_weights[:, j * layer.num_sums_in:(j + 1) * layer.num_sums_in, :, :])
                )

    for column in range(model.num_tasks):
        plt.axvline(x=column, c='lightgrey', linewidth=30)
        if columns and len(columns) > 0 and column > 0:
            plt.annotate(str(columns[column - 1]), (column, -2), ha='center', va='center', size=15)
        plt.plot(column, -1, 'ko', markersize=20)
        plt.plot(column, -2, 'wo', markersize=20)

    for i, column_weights in enumerate(pspn_weights):
        column_weights = torch.tensor(column_weights)
        column_weight_distributions = column_weights / torch.sum(column_weights, dim=1).unsqueeze(1)

        for row, weights in enumerate(column_weight_distributions):
            for column, weight in enumerate(weights):
                plt.plot(column, row, 'ko', markersize=20)
                plt.plot([column, i], [row - 1, row], c='k', alpha=weight.item(), linewidth=weight * 3)


    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument("--load", type=str, default=None, help="Model name to be loaded (default: create new model)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' for gpu, 'cpu' for cpu) (default: cuda)")

    parser.add_argument("--name", type=str, default=None, help="Model name (default: timestamp)")

    parser.add_argument("--test-frequency", type=int, default=10, help="Number of steps between testing during training (default: 10)")
    parser.add_argument("--dataset", type=str, default='mnist', help="Options: mnist, svhn, cifar10 (default: mnist)")

    parser.add_argument("--num", type=int, default=1, help="Number of repetitions (default: 1)")
    parser.add_argument("--task-size", type=int, default=2, help="Number of classes per task (default: 2)")
    parser.add_argument("--starting-task", type=int, default=0, help="Index of first task (default: 0)")
    parser.add_argument("--task-intersection", type=int, default=0, help="Number of classes adopted from previous task (default: 0)")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks (default: 5)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--train-batch-size", type=int, default=128, help="Batch size during training (default: 128)")
    parser.add_argument("--val-batch-size", type=int, default=2048, help="Batch size during validation (default: 512)")
    parser.add_argument('--leaf-search', action='store_true', default=False, help="Toggle the leaf searching (default: off)")
    parser.add_argument('--isolated-column-search', action='store_true', default=False, help="Toggle the isolated column searching (default: off)")
    parser.add_argument('--column-search', action='store_true', default=False, help="Toggle the column searching (default: off)")
    parser.add_argument('--trained-search', action='store_true', default=False, help="Toggle the column searching (default: off)")

    parser.add_argument('--test-spn', action='store_true', default=False, help="Toggle the spn testing of final task (default: off)")

    parser.add_argument("--num-search-batches", type=int, default=128, help="Number of batches to use for the column searching (default: 128)")
    parser.add_argument("--num-training-batches", type=int, default=128, help="Number of training batches to use for the column searching (default: 128)")

    parser.add_argument("--num-sums", type=int, default=3, help="Number of sum nodes (default: 5)")
    parser.add_argument("--num-leaves", type=int, default=3, help="Number of leaf nodes (default: 5)")
    parser.add_argument("--num-repetitions", type=int, default=1, help="Number of repetitions of the network (default: 1)")
    parser.add_argument("--depth", type=int, default=2, help="Depth of the network (default: 3)")

    parser.add_argument("--seed", type=int, default=0, help="Seed used for random generator (default: 0)")

    return parser.parse_args()


def get_datasets(dataset, task_size, task_index, task_intersection, train_batch_size, test_batch_size):
    from_class = task_index * (task_size - task_intersection)
    to_class = from_class + task_size

    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

        train_idx = ((train_dataset.targets >= from_class) & (train_dataset.targets < to_class))
        test_idx = ((test_dataset.targets >= from_class) & (test_dataset.targets < to_class))

        train_dataset.data = train_dataset.data[train_idx]
        test_dataset.data = test_dataset.data[test_idx]

        train_dataset.targets = train_dataset.targets[train_idx] - from_class
        test_dataset.targets = test_dataset.targets[test_idx] - from_class
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transforms.ToTensor())
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())

        train_idx = ((train_dataset.labels >= from_class) & (train_dataset.labels < to_class))
        test_idx = ((test_dataset.labels >= from_class) & (test_dataset.labels < to_class))

        train_dataset.data = train_dataset.data[train_idx]
        test_dataset.data = test_dataset.data[test_idx]

        train_dataset.labels = train_dataset.labels[train_idx] - from_class
        test_dataset.labels = test_dataset.labels[test_idx] - from_class

    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

        train_idx = ((torch.tensor(train_dataset.targets) >= from_class) & (torch.tensor(train_dataset.targets) < to_class))
        test_idx = ((torch.tensor(test_dataset.targets) >= from_class) & (torch.tensor(test_dataset.targets) < to_class))

        train_dataset.data = train_dataset.data[train_idx]
        test_dataset.data = test_dataset.data[test_idx]

        train_dataset.targets = torch.tensor(train_dataset.targets)[train_idx] - from_class
        test_dataset.targets = torch.tensor(test_dataset.targets)[test_idx] - from_class

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


def plotLoss(plt, losses, min_loss, convergence_border):
    plt.plot(losses)

    if convergence_border != len(losses) - 1:
        plt.axhline(y=min_loss, c='lightgrey')
        plt.axvline(x=convergence_border, c='r')
    plt.set_title('loss: ' + str(round(losses[-1], 3)))


def plotAccuracy(plt, accuracies, threshold, classification_border):
    plt.plot(accuracies)
    if classification_border is not None:
        plt.axhline(y=threshold, c='lightgrey')
        plt.axvline(x=classification_border, c='r')
    plt.set_title('acc: ' + str(round(accuracies[-1], 2)))


def evaluateLoss(losses):
    threshold = 0.07

    # Losses
    min_loss = losses[0]
    convergence_border = 0
    for i, loss in enumerate(losses):
        if loss < min_loss - threshold:
            min_loss = loss
            convergence_border = i
    reached_loss = sum(losses[convergence_border:]) / (len(losses) - convergence_border)
    return reached_loss, convergence_border


def evaluateAcc(accuracies):
    threshold = 0.95
    # Accuracies
    classification_border = None
    for i, acc in enumerate(accuracies):
        if acc > threshold:
            classification_border = i
            break
    return threshold, classification_border


def printProgress(time, acc, loss, batch, batches, epoch, epochs, rep, num, task=None, tasks=None):
    if task is None:
        s = int(time * (num * epochs * batches - (rep * epochs * batches + epoch * batches + batch)))
        prog = round((rep * epochs * batches + epoch * batches + batch) / (num * epochs * batches) * 100, 2)
    else:
        s = int(time * (num * tasks * epochs * batches - (
                    rep * tasks * epochs * batches + task * epochs * batches + epoch * batches + batch)))
        prog = round((rep * tasks * epochs * batches + task * epochs * batches + epoch * batches + batch) / (
                    num * tasks * epochs * batches) * 100, 2)
    # else:
    #     s = int(time * (tasks * epochs * batches - (task * epochs * batches + epoch * batches + batch)))
    #     prog = round((task * epochs * batches + epoch * batches + batch) / (tasks * epochs * batches) * 100, 2)
    h = s // 3600
    s = s % 3600
    m = s // 60
    s = s % 60
    itps = round(1 / time, 3)
    acc = round(acc * 100, 2)
    if task is None:
        print("\rRep: {} / {} - Epoch: {} / {} - Batch: {} / {} ( Progress: {}% ) | Time Remaining: {}h :{}m :{}s ({} it/s) | Accuracy: {}% | Loss: {}"
            .format(rep + 1, num, epoch + 1, epochs, batch + 1, batches, prog, h, m, s, itps, acc, loss), end="")
    else:
        print("\rRep: {} / {} - Task: {} / {} - Epoch: {} / {} - Batch: {} / {} ( Progress: {}% ) | Time Remaining: {}h :{}m :{}s ({} it/s) | Accuracy: {}% | Loss: {}"
            .format(rep + 1, num, task + 1, tasks, epoch + 1, epochs, batch + 1, batches, prog, h, m, s, itps, acc, loss), end="")
