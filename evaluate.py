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
#name = "ncs_nti"
#name = "cs-nti"
#name = "ncs_ti"
name = "30r_cs_ti_d3"

intersection_type = 'ni'
search_type = 'tcs'

type = intersection_type + '_' + search_type
name = "300_2t5s_svhn_{}_15_15_3_5".format(type)

plot = True
csv = False

with open('csv/{}.csv'.format(name), 'a') as file:
    if csv:
        file.write('Accuracy (SPN); Accuracy (PSPN); Classification (SPN); Classification (PSPN); Loss (SPN); Loss (PSPN); Convergence (SPN); Convergence (PSPN)\n')

    for index in range(reps):
        if reps == 1:
            file_name = "pspn-backup_{}.pt".format(name)
        else:
            file_name = "pspn-backup_{}_{}.pt".format(name, index)

        spn_file_name = file_name.replace(search_type, "spn")

        checkpoint = torch.load('./model/backup/{}'.format(file_name), map_location=torch.device('cpu'))
        standard = torch.load('./model/backup/{}'.format(spn_file_name), map_location=torch.device('cpu'))

        config = checkpoint['config']
        pspn = PSPN(config, checkpoint['task_progress'] + 1)
        pspn.load_state_dict(checkpoint['pspn_state_dict'])

        if plot:
            columns = None
            if 'columns' in checkpoint:
                columns = checkpoint['columns']
            analyseWeights(pspn, columns)

        losses = checkpoint['losses']
        #losses[-1] = standard['losses'][-1]
        losses.append(standard['losses'][-1])

        accuracies = checkpoint['accuracies']
        #accuracies[-1] = standard['accuracies'][-1]
        accuracies.append(standard['accuracies'][-1])

        num_epochs = checkpoint['num_epochs']
        num_tasks = checkpoint['num_tasks'] - checkpoint['starting_task']

        classification_borders = []
        convergence_borders = []
        if len(losses) == 1:
            fig, ax = plt.subplots(1, 1, sharex=True)
            min_loss, convergence_border = evaluateLoss(losses[0])
            if plot:
                plotLoss(ax, losses[0], min_loss, convergence_border)
                plt.show()

            fig, ax = plt.subplots(1, 1, sharex=True)
            threshold, classification_border = evaluateAcc(accuracies[0])
            if plot:
                plotAccuracy(ax, accuracies[0], threshold, classification_border)
                plt.show()
        else:
            if plot:
                fig, axs = plt.subplots(len(losses) // 2, 2 + (len(losses) % 2), sharex=True)
                for i, ax in enumerate(axs.reshape(-1)):
                    min_loss, convergence_border = evaluateLoss(losses[i])
                    convergence_borders.append(convergence_border)
                    plotLoss(ax, losses[i], min_loss, convergence_border)
                plt.show()

                fig, axs = plt.subplots(len(losses) // 2, 2 + len(losses) % 2, sharex=True)
                for i, ax in enumerate(axs.reshape(-1)):
                    threshold, classification_border = evaluateAcc(accuracies[i])
                    classification_borders.append(classification_border)
                    plotAccuracy(ax, accuracies[i], threshold, classification_border)
                plt.show()
            elif csv:
                for i in range(len(losses)):
                    min_loss, convergence_border = evaluateLoss(losses[i])
                    convergence_borders.append(convergence_border)

                    threshold, classification_border = evaluateAcc(accuracies[i])
                    classification_borders.append(classification_border)

        if csv:
            file.write(str(round(accuracies[-1][-1], 3)).replace(".",",") + ";" + str(round(accuracies[-2][-1], 3)).replace(".",",") + ";" + \
                       str(classification_borders[-1]) + ';' + str(classification_borders[-2]) + ';' + \
                       str(round(losses[-1][-1], 5)).replace(".",",") + ';' + str(round(losses[-2][-1], 5)).replace(".",",") + ';' + \
                       str(convergence_borders[-1]) + ';' + str(convergence_borders[-2]) + '\n')

        if reps > 1 and plot: input("Press Enter")

