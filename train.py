import os
import torch
from dataset import Dataset, get_transform

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

print(device)

# Data paths
cwd = os.getcwd()
img_dir = os.path.join(cwd, 'data', '2020intent', 'images', 'low')
train_annotation_path = os.path.join(cwd, 'data', '2020intent', 'annotations', 'intentonomy_train2020.json')
val_annotation_path = os.path.join(cwd, 'data', '2020intent', 'annotations', 'intentonomy_val2020.json')

# Parameters
params = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 6}
max_epochs = 100

train_dataset = Dataset(img_dir, train_annotation_path, get_transform(), type='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, **params)

val_dataset = Dataset(img_dir, val_annotation_path, get_transform(), type='val')
val_dataloader = torch.utils.data.DataLoader(val_dataset, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in train_dataloader:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        
        # Model computations

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in val_dataloader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations