from matplotlib import pyplot as plt

from helper import analyseWeights
from helper import evaluateAcc
from helper import evaluateLoss
from helper import plotAccuracy
from helper import plotLoss

import torch

from simple_einet.distributions.normal import Normal
from simple_einet.einet import PSPN, EinetColumnConfig

# config = EinetColumnConfig(
#     num_channels=3,
#     num_features=32*32,
#     num_sums=15,
#     num_leaves=15,
#     num_repetitions=3,
#     depth=5,
#     leaf_type=Normal,
#     leaf_kwargs={},
#     num_classes=5,
#     seed=0,
# )
# pspn = PSPN(config)
# pspn.expand()
# pspn.expand()
# analyseWeights(pspn)
#

reps = 1

epochs = '50'
tasks = '3t3s'
dataset = 'svhn'
intersection_type = 'ni'
search_type = 'fcs'
column_param = '10_10_2_4'

name = '_'.join((epochs, tasks, dataset, intersection_type, search_type, column_param))

plot = True
csv = False

with open('csv/{}.csv'.format(name), 'a') as file:
    if csv:
        file.write('Accuracy (SPN); Accuracy (PSPN); Classification (SPN); Classification (PSPN); Loss (SPN); Loss (PSPN); Convergence (SPN); Convergence (PSPN)\n')

    for index in range(reps):
        if reps == 1:
            pspn_name = "pspn_{}.pt".format(name)
            spn_name = "spn_{}.pt".format(name).replace(search_type, "ns")
        else:
            file_name = "pspn-backup_{}_{}.pt".format(name, index)

        pspn_checkpoint = torch.load('./model/backup/{}'.format(pspn_name), map_location=torch.device('cpu'))
        spn_checkpoint = torch.load('./model/backup/{}'.format(spn_name), map_location=torch.device('cpu'))

        config = pspn_checkpoint['config']
        pspn = PSPN(config, pspn_checkpoint['task_progress'] + 1)
        pspn.load_state_dict(pspn_checkpoint['pspn_state_dict'])

        if plot:
            columns = None
            if 'columns' in pspn_checkpoint:
                columns = pspn_checkpoint['columns']
            analyseWeights(pspn, columns, name='{} - {}'.format(intersection_type, search_type).upper())

        pspn_losses = pspn_checkpoint['losses']
        spn_losses = spn_checkpoint['losses']

        pspn_accuracies = pspn_checkpoint['accuracies']
        spn_accuracies = spn_checkpoint['accuracies']

        num_epochs = pspn_checkpoint['num_epochs']
        num_tasks = pspn_checkpoint['num_tasks'] - pspn_checkpoint['starting_task']

        pspn_classification_borders = []
        pspn_convergence_borders = []
        spn_classification_borders = []
        spn_convergence_borders = []
        for i, pspn_loss in enumerate(pspn_losses):
            pspn_min_loss, pspn_convergence_border = evaluateLoss(pspn_loss)
            spn_min_loss, spn_convergence_border = evaluateLoss(spn_losses[i])
            if plot:
                plotLoss(plt, pspn_loss, pspn_min_loss, pspn_convergence_border, c='blue', ac='lightblue', label='PSPN')
                plotLoss(plt, spn_losses[i], spn_min_loss, spn_convergence_border, c='red', ac='pink', label='SPN')
                plt.legend(loc="upper left")
                plt.title('Loss: Column {}'.format(i))
                plt.show()
            if csv:
                pspn_convergence_borders.append(pspn_convergence_border)
                spn_convergence_borders.append(spn_convergence_border)

            pspn_threshold, pspn_classification_border = evaluateAcc(pspn_accuracies[i])
            spn_threshold, spn_classification_border = evaluateAcc(spn_accuracies[i])
            if plot:
                plotAccuracy(plt, pspn_accuracies[i], pspn_threshold, pspn_classification_border, c='blue', ac='lightblue', label='PSPN')
                plotAccuracy(plt, spn_accuracies[i], spn_threshold, spn_classification_border, c='red', ac='pink', label='SPN')
                plt.legend(loc="upper left")
                plt.title('Accuracy: Column {}'.format(i))
                plt.show()
            if csv:
                pspn_classification_borders.append(pspn_convergence_border)
                spn_classification_borders.append(spn_convergence_border)

        if csv:
            file.write(str(round(spn_accuracies[-1][-1], 3)).replace(".",",") + ";" + str(round(pspn_accuracies[-1][-1], 3)).replace(".",",") + ";" + \
                       str(spn_classification_borders[-1]) + ';' + str(pspn_classification_borders[-1]) + ';' + \
                       str(round(spn_losses[-1][-1], 5)).replace(".",",") + ';' + str(round(pspn_losses[-1][-1], 5)).replace(".",",") + ';' + \
                       str(pspn_convergence_borders[-1]) + ';' + str(spn_convergence_borders[-1]) + '\n')

        if reps > 1 and plot: input("Press Enter")

