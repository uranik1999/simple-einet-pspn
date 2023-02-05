from helper import evaluate

import torch

# Load Model
folder_name = "backup/"
file_name = "pspn-backup_1675591142.pt"

checkpoint = torch.load('./model/{}{}'.format(folder_name, file_name), map_location=torch.device('cpu'))

losses = checkpoint['losses']
accuracies = checkpoint['accuracies']

num_epochs = checkpoint['num_epochs']
num_tasks = checkpoint['num_tasks'] # - checkpoint['starting_task']

evaluate(losses, accuracies, num_epochs, num_tasks)