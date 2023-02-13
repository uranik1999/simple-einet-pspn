from matplotlib import pyplot as plt

import helper

import torch


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
    threshold = 0.03

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


reps = 10
name = "ncs-1nti"

for index in range(reps):
    if reps == 1:
        file_name = "pspn-backup_{}.pt".format(name)
    else:
        file_name = "pspn-backup_{}_{}.pt".format(name, index)

    checkpoint = torch.load('./model/backup/{}'.format(file_name), map_location=torch.device('cpu'))

    losses = checkpoint['losses']
    accuracies = checkpoint['accuracies']

    num_epochs = checkpoint['num_epochs']
    num_tasks = checkpoint['num_tasks'] - checkpoint['starting_task']


    fig, axs = plt.subplots(len(losses) // 2, 2 + len(losses) % 2, sharex=True)
    for i, ax in enumerate(axs.reshape(-1)):
        min_loss, convergence_border = evaluateLoss(losses[i])
        plotLoss(ax, losses[i], min_loss, convergence_border)

    plt.show()

    fig, axs = plt.subplots(len(losses) // 2, 2 + len(losses) % 2, sharex=True)
    for i, ax in enumerate(axs.reshape(-1)):
        threshold, classification_border = evaluateAcc(accuracies[i])
        plotAccuracy(ax, accuracies[i], threshold, classification_border)

    plt.show()

    input("Press Enter")

