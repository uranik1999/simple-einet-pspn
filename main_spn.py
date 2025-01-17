#!/usr/bin/env python3
from matplotlib import pyplot as plt

from helper import get_datasets
from helper import test
from helper import printProgress
from helper import parse_args

import torch
import torch.nn as nn

from simple_einet.distributions.normal import Normal
from simple_einet.einet import EinetColumnConfig, EinetColumn

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
            task_intersection = args.task_intersection
            num_tasks = args.num_tasks
            num_epochs = args.num_epochs
            lr = args.lr
            train_batch_size = args.train_batch_size
            test_batch_size = args.val_batch_size
            leaf_search = args.leaf_search
            isolated_column_search = args.isolated_column_search
            column_search = args.column_search
            trained_search = args.trained_search
            num_search_batches = args.num_search_batches
            num_training_epochs = args.num_training_epochs
            test_frequency = args.test_frequency

            dataset = args.dataset

            # Progress
            losses = []
            accuracies = []
            columns = []
            column_losses = []

            epoch_progress = 0
            task_progress = starting_task

            if dataset == 'mnist':
                num_features = 28 * 28
                num_channels = 1
            else:
                num_features = 32 * 32
                num_channels = 3
            # Model parameters
            config = EinetColumnConfig(
                num_channels=num_channels,
                num_features=num_features,
                num_sums=args.num_sums,
                num_leaves=args.num_leaves,
                num_repetitions=args.num_repetitions,
                depth=args.depth,
                leaf_type=Normal,
                leaf_kwargs={},
                num_classes=args.task_size,
                seed=args.seed,
            )
            spn = EinetColumn(config, column_index=0).to(device)  # column_index = 0 to exclude lateral connections
            spn_state_dicts = []

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
            leaf_search = checkpoint['leaf_search']
            isolated_column_search = checkpoint['isolated_column_search']
            column_search = checkpoint['column_search']
            trained_search = checkpoint['trained_search']
            num_search_batches = checkpoint['num_search_batches']
            num_training_epochs = checkpoint['num_training_epochs']
            test_frequency = checkpoint['test_frequency']

            dataset = checkpoint['dataset']

            # Load progress
            losses = checkpoint['losses']
            accuracies = checkpoint['accuracies']
            columns = checkpoint['columns']
            column_losses = checkpoint['column_losses']

            epoch_progress = checkpoint['epoch_progress']
            task_progress = checkpoint['task_progress']

            # Load model
            config = checkpoint['config']
            spn = EinetColumn(config, column_index=0).to(device)  # column_index = 0 to exclude lateral connections

            spn_state_dicts = checkpoint['spn_state_dicts']
            spn.load_state_dict(spn_state_dicts[-1])

            # Load optimizer
            loss = checkpoint['loss']

        previous_columns = []
        for task in range(task_progress, num_tasks):
            # Get training data for current task
            train_dataloader, test_dataloader = get_datasets(dataset, task_size, task, task_intersection, train_batch_size, test_batch_size)
            batches = len(train_dataloader)
            test_data, test_labels = next(iter(test_dataloader))
            test_data, test_labels = test_data.to(device), test_labels.to(device)

            if epoch_progress == 0:
                spn = EinetColumn(config, column_index=0).to(device)
                previous_columns.append(spn)

                if column_search or leaf_search or isolated_column_search:
                    print("Searching Columns: ", end="")
                    column_index, mean_losses = columnSearch(device, previous_columns, spn, config, train_dataloader,
                                                                 num_search_batches, loss, leaf_search,
                                                                 isolated_column_search, column_search, trained_search,
                                                                 num_training_epochs, lr, task_size)
                    columns.append(column_index)
                    column_losses.append(mean_losses)

                losses.append([])
                accuracies.append([])

            optimizer = torch.optim.Adam(spn.parameters(), lr=lr)

            for epoch in range(epoch_progress, num_epochs):
                t = time.time()

                for batch, (data, labels) in enumerate(train_dataloader):

                    data = data.to(device)
                    labels = labels.to(device)

                    # Training
                    optimizer.zero_grad()
                    likelihood, _ = spn(data, prev_column_outputs=[])
                    prior = torch.log(torch.tensor(1 / task_size))  # p(y)
                    marginal = (likelihood + prior).logsumexp(-1).unsqueeze(1)  # p(x) = sum(p(x, y)) = sum(p(x|y) * p(y))
                    posterior = likelihood + prior - marginal  # p(y|x) = p(x|y) * p(y) / p(x)
                    err = loss(posterior, labels)
                    err.backward()
                    optimizer.step()

                    # if batch % test_frequency == 0:
                    #     losses[-1].append(err.item())
                    #     accuracies[-1].append(test(pspn, test_data, test_labels))
                    #
                    #     t = time.time() - t
                    #
                    #     printProgress(t, accuracies[-1][-1], losses[-1][-1], batch, batches, epoch, num_epochs, rep, num, task, num_tasks)
                t = time.time() - t

                losses[-1].append(err.item())
                accuracies[-1].append(test(spn, test_data, test_labels, task_size))

                printProgress(t, accuracies[-1][-1], losses[-1][-1], batch, batches, epoch, num_epochs, rep, num, task, num_tasks)

                spn_state_dicts.append(spn.state_dict())
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
                    'leaf_search': leaf_search,
                    'isolated_column_search': isolated_column_search,
                    'column_search': column_search,
                    'trained_search': trained_search,
                    'num_search_batches': num_search_batches,
                    'num_training_epochs': num_training_epochs,
                    'test_frequency': test_frequency,

                    'dataset': dataset,

                    'config': config,
                    'spn_state_dicts': spn_state_dicts,

                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,

                    'losses': losses,
                    'accuracies': accuracies,
                    'columns': columns,
                    'column_losses': column_losses,

                    'epoch_progress': epoch + 1,
                    'task_progress': task
                }, './model/backup/spn_{}.pt'.format(model_name))

            print()
            epoch_progress = 0
            previous_columns[-1] = spn


def columnSearch(device, previous_columns, model, column_config, dataloader, nr_search_batches, loss, leaf_search, isolated_column_search, column_search, trained_search, nr_training_epochs, lr, task_size):
    with torch.no_grad():

        test_model = EinetColumn(column_config, column_index=0).to(device) # column_index = 0 to exclude lateral connections

        mean_losses = []
        mean_norms = []
        for column_index, column in enumerate(reversed(previous_columns)):
            print("\rSearching Columns: Building Column {}".format(column_index), end="")
            if column_search:
                column_state_dict = column.state_dict()
                test_model.load_state_dict(column_state_dict)
            elif leaf_search:
                test_model.leaf.load_state_dict(column.leaf.state_dict())

            if trained_search:
                with torch.enable_grad():

                    optimizer = torch.optim.Adam(test_model.parameters(), lr=lr)

                    for epoch in range(nr_training_epochs):
                        for batch, (data, labels) in enumerate(dataloader):
                            data = data.to(device)
                            labels = labels.to(device)

                            likelihood, _ = test_model(data, prev_column_outputs=[])

                            prior = torch.log(torch.tensor(1 / task_size))  # p(y)
                            marginal = (likelihood + prior).logsumexp(-1).unsqueeze(
                                1)  # p(x) = sum(p(x, y)) = sum(p(x|y) * p(y))
                            posterior = likelihood + prior - marginal  # p(y|x) = p(x|y) * p(y) / p(x)

                            err = loss(posterior, labels)
                            err.backward()
                            optimizer.step()

                        print("\rSearching Columns: Training Column {} - Epoch {} / {} - Loss {}".format(column_index, epoch + 1, nr_training_epochs, err.item()), end="")

            print()

            with torch.enable_grad():

                optimizer = torch.optim.Adam(test_model.parameters(), lr=lr)
                losses = []
                norms = []
                for batch, (data, labels) in enumerate(dataloader):
                    if batch >= nr_search_batches:
                        break

                    data = data.to(device)
                    labels = labels.to(device)


                    optimizer.zero_grad()

                    likelihood, _ = test_model(data, prev_column_outputs=[])

                    prior = torch.log(torch.tensor(1 / task_size))  # p(y)
                    marginal = (likelihood + prior).logsumexp(-1).unsqueeze(1)  # p(x) = sum(p(x, y)) = sum(p(x|y) * p(y))
                    posterior = likelihood + prior - marginal  # p(y|x) = p(x|y) * p(y) / p(x)

                    err = loss(posterior, labels)
                    err.backward()

                    grad_norms = []
                    for param in test_model.parameters():
                        if param.requires_grad:
                            grad_norms.append(param.grad.norm())

                    norms.append(sum(grad_norms) / len(grad_norms))
                    losses.append(err)

                    print("\rSearching Columns: Testing Column {} - Batch {} / {} - Loss {} - Norm {}".format(column_index,
                                                                                                    batch,
                                                                                                    nr_training_epochs,
                                                                                                    losses[-1], norms[-1]), end="")

            mean_loss = sum(losses) / len(losses)
            mean_losses.append(mean_loss)

            mean_norm = sum(norms) / len(norms)
            mean_norms.append(mean_loss)

            print("\rSearching Columns: Tested Column {} - Mean Loss {} - Mean Norm {}".format(column.column_index, mean_loss, mean_norm), end="")
            print()

        mean_losses = list(reversed(mean_losses))
        column_index = mean_losses.index(min(mean_losses))
        # column_index = 0
        # mean_norms = list(reversed(mean_norms))
        # column_index = mean_norms.index(min(mean_norms))

        print("\rSearching Columns: Copying Column: {}".format(column_index), end="")

        if column_search:
            column = previous_columns[column_index]
            column_state_dict = column.state_dict()
            model.load_state_dict(column_state_dict)
        elif leaf_search:
            model.leaf.load_state_dict(previous_columns[column_index].leaf.state_dict())

        print()
    return column_index, mean_losses



if __name__ == "__main__":
    main()
