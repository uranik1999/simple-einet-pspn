from matplotlib import pyplot as plt

from helper import evaluate

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
    epsilon = 0.15

    # Losses
    min_loss = losses[0]
    convergence_border = 0
    for i, loss in enumerate(losses):
        if loss < min_loss * (1 + epsilon):
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


reps = 1

for index in range(reps):
    folder_name = "backup/"

    # pspn
    file_name = "pspn-backup_pspn_{}.pt".format(index)

    checkpoint = torch.load('./model/{}{}'.format(folder_name, file_name), map_location=torch.device('cpu'))

    pspn_losses = checkpoint['losses']
    pspn_accuracies = checkpoint['accuracies']

    pspn_num_epochs = checkpoint['num_epochs']
    pspn_num_tasks = checkpoint['num_tasks'] - checkpoint['starting_task']


    # spn
    file_name = "pspn-backup_spn_{}.pt".format(index)

    checkpoint = torch.load('./model/{}{}'.format(folder_name, file_name), map_location=torch.device('cpu'))

    spn_losses = checkpoint['losses']
    spn_accuracies = checkpoint['accuracies']

    spn_num_epochs = checkpoint['num_epochs']
    spn_num_tasks = checkpoint['num_tasks'] - checkpoint['starting_task']

    losses = pspn_losses + spn_losses
    accuracies = pspn_accuracies + spn_accuracies

    fig, axs = plt.subplots(3, 2, sharex=True)
    for i, ax in enumerate(axs.reshape(-1)):
        min_loss, convergence_border = evaluateLoss(losses[i])
        plotLoss(ax, losses[i], min_loss, convergence_border)

    plt.show()

    fig, axs = plt.subplots(3, 2, sharex=True)
    for i, ax in enumerate(axs.reshape(-1)):
        threshold, classification_border = evaluateAcc(accuracies[i])
        plotAccuracy(ax, accuracies[i], threshold, classification_border)

    plt.show()

    input("Press Enter")

