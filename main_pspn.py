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
from simple_einet.einet import PSPN, EinetColumnConfig

import time


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
            num_tasks = args.num_tasks
            num_epochs = args.num_epochs
            lr = args.lr
            train_batch_size = args.train_batch_size
            test_batch_size = args.val_batch_size

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

            loss = nn.NLLLoss()
        else:
            model_name = args.load

            # Select model
            checkpoint = torch.load('./model/backup/pspn-backup_{}.pt'.format(args.load))

            # Load training parameters
            lr = checkpoint['lr']
            task_size = checkpoint['task_size']
            starting_task = checkpoint['starting_task']
            num_tasks = checkpoint['num_tasks']
            num_epochs = checkpoint['num_epochs']
            train_batch_size = checkpoint['train_batch_size']
            test_batch_size = checkpoint['test_batch_size']

            # Load progress
            losses = checkpoint['losses']
            accuracies = checkpoint['accuracies']

            epoch_progress = checkpoint['epoch_progress']
            task_progress = checkpoint['task_progress']

            # Load model
            config = checkpoint['config']
            pspn = PSPN(config, task_progress + 1)
            pspn.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer
            loss = checkpoint['loss']

        for task in range(task_progress, num_tasks):
            # Get training data for current task
            train_dataloader, test_dataloader = get_datasets(task_size, task, train_batch_size, test_batch_size)
            batches = len(train_dataloader)
            test_data, test_labels = next(iter(test_dataloader))
            test_data, test_labels = test_data.to(device), test_labels.to(device)

            if epoch_progress == 0:
                pspn.expand()
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

                    printProgress(t, accuracies[-1][-1], losses[-1][-1], batch, batches, epoch, num_epochs, task, num_tasks, rep, num)

                torch.save({
                    'num': num,
                    'task_size': task_size,
                    'starting_task': starting_task,
                    'num_tasks': num_tasks,
                    'num_epochs': num_epochs,
                    'lr': lr,
                    'train_batch_size': train_batch_size,
                    'test_batch_size': train_batch_size,

                    'config': config,
                    'model_state_dict': pspn.state_dict(),

                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,

                    'losses': losses,
                    'accuracies': accuracies,

                    'epoch_progress': epoch + 1,
                    'task_progress': task
                }, './model/backup/pspn-backup_{}.pt'.format(model_name))

            epoch_progress = 0


if __name__ == "__main__":
    main()
