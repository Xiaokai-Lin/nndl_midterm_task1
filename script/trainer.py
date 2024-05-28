import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split
import lightning as L
import numpy as np
from data import Cub2011
from model import CubClassifier

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--workpath", type=str, default="~/sms2/nndl/midterm/")
parser.add_argument("--save_name", type=str, default="cub_classifier")
parser.add_argument("--pretrained", type=bool, default=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--optimizer_name", type=str, default="SGD")
parser.add_argument("--lr", type=float, default=0.1)

args = parser.parse_args()
# Set random seed for reproducibility
torch.manual_seed(42)

# Define transformation for the dataset
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the CUB-2011 dataset
train_datasest = Cub2011(root=args.workpath, download=True, transform=transform_train)
test_dataset = Cub2011(root=args.workpath, download=False, train=False, transform=transform_test)

# Stratified Sampling for train and val
train_idx, validation_idx = train_test_split(np.arange(len(train_datasest)),
                                             test_size=0.1,
                                             random_state=999,
                                             shuffle=True,
                                             stratify=train_datasest.data['target'].to_numpy())

# Subset dataset for train and val
validation_dataset = torch.utils.data.Subset(train_datasest, validation_idx)
train_dataset = torch.utils.data.Subset(train_datasest, train_idx)



# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

CHECKPOINT_PATH = '~/sms2/nndl/midterm/'
def train_model(workpath, save_name, pretrained, epochs, optimizer_name, lr):
    """Train model.

    Args:
        model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
    """

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(workpath, save_name),
        # We run on a single GPU (if possible)
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        callbacks=[
            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc", filename=save_name
            ),  
            LearningRateMonitor("epoch"),
        ],
    )
    
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(workpath, save_name + '/lightning_logs/version_0/checkpoints/', save_name  + ".ckpt")
    print(pretrained_filename)
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = CubClassifier.load_from_checkpoint(pretrained_filename)
    else:
        # To be reproducible
        L.seed_everything(42)
        model = CubClassifier(pretrained = pretrained, optimizer_name=optimizer_name, lr=lr)
        trainer.fit(model, train_loader, val_loader)
        
        # Load best checkpoint after training
        model = CubClassifier.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    print(f"Test accuracy: {result['test']:.4f}, Validation accuracy: {result['val']:.4f}")
    print("Model checkpointed at:", trainer.checkpoint_callback.best_model_path)

# Parse the user inputs and defaults (returns a argparse.Namespace)
train_model(args.workpath, args.save_name, args.pretrained, args.epochs, args.optimizer_name, args.lr)
