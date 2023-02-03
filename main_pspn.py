#!/usr/bin/env python3
from helper import get_datasets
from helper import test
from helper import evaluate
from helper import printProgress

import torch
import torch.nn as nn

from simple_einet.distributions.normal import Normal
from simple_einet.einet import PSPN, EinetColumnConfig

import time


def main():
    torch.manual_seed(0)

    timestamp = 0
    load_model = False
    if load_model:
        # Select model
        checkpoint = torch.load('./model/backup/pspn-backup_{}.pt'.format(timestamp))

        # Load training parameters
        lr = checkpoint['lr']
        task_size = checkpoint['task_size']
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
    else:
        timestamp = int(timestamp)

        # Training parameters
        lr = 0.002
        task_size = 2
        num_tasks = 5
        num_epochs = 100
        train_batch_size = 200
        test_batch_size = 500

        # Progress
        losses = []
        accuracies = []

        epoch_progress = 0
        task_progress = 0

        # Model parameters
        config = EinetColumnConfig(
            num_channels=1,
            num_features=28 * 28,
            num_sums=10,
            num_leaves=10,
            num_repetitions=3,
            depth=4,
            leaf_type=Normal,
            leaf_kwargs={},
            num_classes=2,
            seed=0,
        )
        pspn = PSPN(config)

        loss = nn.NLLLoss()

    for task in range(task_progress, num_tasks):
        # Get training data for current task
        train_dataloader, test_dataloader = get_datasets(task_size, task, train_batch_size, test_batch_size)
        batches = len(train_dataloader)
        test_data, test_labels = next(iter(test_dataloader))

        if epoch_progress == 0:
            pspn.expand()

        optimizer = torch.optim.Adam(pspn.parameters(), lr=lr)

        for epoch in range(epoch_progress, num_epochs):
            for batch, (data, labels) in enumerate(train_dataloader):
                t = time.time()

                # Training
                optimizer.zero_grad()
                pred = pspn(data)
                err = loss(pred, labels)
                err.backward()
                optimizer.step()

                losses.append(err.item())
                accuracies.append(test(pspn, test_data, test_labels))

                t = time.time() - t

                printProgress(t, accuracies[-1], losses[-1], batch, batches, epoch, num_epochs, task, num_tasks)

            torch.save({
                'lr': lr,
                'task_size': task_size,
                'num_tasks': num_tasks,
                'num_epochs': num_epochs,
                'train_batch_size': train_batch_size,
                'test_batch_size': train_batch_size,

                'config': config,
                'model_state_dict': pspn.state_dict(),

                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,

                'losses': losses,
                'accuracies': accuracies,

                'epoch_progress': epoch,
                'task_progress': task
            }, './model/backup/pspn-backup_{}.pt'.format(int(timestamp)))

        epoch_progress = 0


if __name__ == "__main__":
    main()
