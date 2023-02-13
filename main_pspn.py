#!/usr/bin/env python3
from matplotlib import pyplot as plt

from helper import get_datasets
from helper import test
from helper import evaluate
from helper import printProgress
from helper import parse_args

import torch
import torch.nn as nn

from simple_einet.distributions.normal import Normal
from simple_einet.einet import PSPN, EinetColumnConfig, EinetColumn

import time


def columnSearch(model, column_config, dataloader, nr_search_batches, loss):
    with torch.no_grad():

        test_column = EinetColumn(column_config, column_index=0)  # column_index = 0 to exclude lateral connections

        mean_losses = []
        for column in reversed(model.columns):
            test_column.leaf.load_state_dict(column.leaf.state_dict())

            losses = []
            for batch, (data, labels) in enumerate(dataloader):
                if batch >= nr_search_batches:
                    break
                likelihood, _ = test_column(data, prev_column_outputs=[])
                prior = -0.6931471805599453  # log(0.5) = p(y)
                marginal = (likelihood + prior).logsumexp(-1).unsqueeze(1)  # p(x) = sum(p(x, y)) = sum(p(x|y) * p(y))
                posterior = likelihood + prior - marginal  # p(y|x) = p(x|y) * p(y) / p(x)
                losses.append(loss(posterior, labels))

            mean_loss = sum(losses) / len(losses)
            mean_losses.append(mean_loss)

        mean_losses = list(reversed(mean_losses))

    return mean_losses.index(min(mean_losses))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    torch.manual_seed(args.seed)
    load = args.load
    num = args.num

    if load is not None:
        num_repetitions = 1

    for rep in range(num):
        if load is None:
            model_name = (args.name if args.name is not None else str(int(time.time()))) + ('_' + str(rep) if num > 1 else '')

            # Training parameters
            num = num
            task_size = args.task_size
            starting_task = args.starting_task
            task_intersection = args.task_intersection
            num_tasks = args.num_tasks
            num_epochs = args.num_epochs
            lr = args.lr
            train_batch_size = args.train_batch_size
            test_batch_size = args.val_batch_size
            column_search = args.column_search
            num_search_batches = args.num_search_batches

            # Progress
            losses = []
            accuracies = []

            epoch_progress = 0
            task_progress = starting_task

            # Model parameters
            config = EinetColumnConfig(
                num_channels=1,
                num_features=28 * 28,
                num_sums=args.num_sums,
                num_leaves=args.num_leaves,
                num_repetitions=args.num_repetitions,
                depth=args.depth,
                leaf_type=Normal,
                leaf_kwargs={},
                num_classes=args.task_size,
                seed=args.seed,
            )
            pspn = PSPN(config).to(device)
            spn = EinetColumn(config, column_index=0).to(device)  # column_index = 0 to exclude lateral connections
            test_spn = args.test_spn

            loss = nn.NLLLoss()
        else:
            model_name = args.load

            # Select model
            checkpoint = torch.load('./model/backup/pspn-backup_{}.pt'.format(args.load))

            # Load training parameters
            lr = checkpoint['lr']
            task_size = checkpoint['task_size']
            starting_task = checkpoint['starting_task']
            task_intersection = checkpoint['task_intersection']
            num_tasks = checkpoint['num_tasks']
            num_epochs = checkpoint['num_epochs']
            train_batch_size = checkpoint['train_batch_size']
            test_batch_size = checkpoint['test_batch_size']
            column_search = checkpoint['column_search']
            num_search_batches = checkpoint['num_search_batches']

            # Load progress
            losses = checkpoint['losses']
            accuracies = checkpoint['accuracies']

            epoch_progress = checkpoint['epoch_progress']
            task_progress = checkpoint['task_progress']

            # Load model
            config = checkpoint['config']
            pspn = PSPN(config, task_progress + 1)
            pspn.load_state_dict(checkpoint['pspn_state_dict'])
            spn = EinetColumn(config, column_index=0).to(device)  # column_index = 0 to exclude lateral connections
            spn.load_state_dict(checkpoint['spn_state_dict'])
            test_spn = checkpoint['test_spn']

            # Load optimizer
            loss = checkpoint['loss']

        for task in range(task_progress, num_tasks):
            # Get training data for current task
            train_dataloader, test_dataloader = get_datasets(task_size, task, task_intersection, train_batch_size, test_batch_size)
            batches = len(train_dataloader)
            test_data, test_labels = next(iter(test_dataloader))
            test_data, test_labels = test_data.to(device), test_labels.to(device)

            if epoch_progress == 0:
                pspn.expand()
                if column_search and task != 0:
                    column_index = columnSearch(pspn, config, train_dataloader, num_search_batches, loss)
                    pspn.columns[-1].leaf.load_state_dict(pspn.columns[column_index].leaf.state_dict())

                losses.append([])
                accuracies.append([])

            pspn = pspn.to(device)
            optimizer = torch.optim.Adam(pspn.parameters(), lr=lr)

            for epoch in range(epoch_progress, num_epochs):
                for batch, (data, labels) in enumerate(train_dataloader):
                    t = time.time()

                    data = data.to(device)
                    labels = labels.to(device)

                    # Training
                    optimizer.zero_grad()
                    likelihood = pspn(data)
                    prior = -0.6931471805599453  # log(0.5) = p(y)
                    marginal = (likelihood + prior).logsumexp(-1).unsqueeze(1)  # p(x) = sum(p(x, y)) = sum(p(x|y) * p(y))
                    posterior = likelihood + prior - marginal  # p(y|x) = p(x|y) * p(y) / p(x)
                    err = loss(posterior, labels)
                    err.backward()
                    optimizer.step()

                    losses[-1].append(err.item())
                    accuracies[-1].append(test(pspn, test_data, test_labels))

                    t = time.time() - t

                    printProgress(t, accuracies[-1][-1], losses[-1][-1], batch, batches, epoch, num_epochs, rep, num, task, num_tasks)

                torch.save({
                    'num': num,
                    'task_size': task_size,
                    'starting_task': starting_task,
                    'task_intersection': task_intersection,
                    'num_tasks': num_tasks,
                    'num_epochs': num_epochs,
                    'lr': lr,
                    'train_batch_size': train_batch_size,
                    'test_batch_size': train_batch_size,
                    'column_search': column_search,
                    'num_search_batches': num_search_batches,

                    'config': config,
                    'pspn_state_dict': pspn.state_dict(),
                    'spn_state_dict': spn.state_dict(),
                    'test_spn': test_spn,

                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,

                    'losses': losses,
                    'accuracies': accuracies,

                    'epoch_progress': epoch + 1,
                    'task_progress': task
                }, './model/backup/pspn-backup_{}.pt'.format(model_name))

            epoch_progress = 0

        if test_spn:
            print()

            if epoch_progress == 0:
                losses.append([])
                accuracies.append([])

            optimizer = torch.optim.Adam(spn.parameters(), lr=lr)

            train_dataloader, test_dataloader = get_datasets(task_size, num_tasks - 1, task_intersection, train_batch_size, test_batch_size)
            batches = len(train_dataloader)
            test_data, test_labels = next(iter(test_dataloader))
            test_data, test_labels = test_data.to(device), test_labels.to(device)

            for epoch in range(epoch_progress, num_epochs):
                for batch, (data, labels) in enumerate(train_dataloader):
                    t = time.time()

                    data = data.to(device)
                    labels = labels.to(device)

                    # Training
                    optimizer.zero_grad()
                    likelihood, _ = spn(data, prev_column_outputs=[])
                    prior = -0.6931471805599453  # log(0.5) = p(y)
                    marginal = (likelihood + prior).logsumexp(-1).unsqueeze(1)  # p(x) = sum(p(x, y)) = sum(p(x|y) * p(y))
                    posterior = likelihood + prior - marginal  # p(y|x) = p(x|y) * p(y) / p(x)
                    err = loss(posterior, labels)
                    err.backward()
                    optimizer.step()

                    losses[-1].append(err.item())
                    accuracies[-1].append(test(pspn, test_data, test_labels))

                    t = time.time() - t

                    printProgress(t, accuracies[-1][-1], losses[-1][-1], batch, batches, epoch, num_epochs, rep, num)

                torch.save({
                    'num': num,
                    'task_size': task_size,
                    'starting_task': starting_task,
                    'task_intersection': task_intersection,
                    'num_tasks': num_tasks,
                    'num_epochs': num_epochs,
                    'lr': lr,
                    'train_batch_size': train_batch_size,
                    'test_batch_size': train_batch_size,
                    'column_search': column_search,
                    'num_search_batches': num_search_batches,

                    'config': config,
                    'pspn_state_dict': pspn.state_dict(),
                    'spn_state_dict': spn.state_dict(),
                    'test_spn': test_spn,

                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,

                    'losses': losses,
                    'accuracies': accuracies,

                    'epoch_progress': epoch + 1,
                    'task_progress': task
                }, './model/backup/pspn-backup_{}.pt'.format(model_name))

            print('\n' + model_name)
            print()


if __name__ == "__main__":
    main()
