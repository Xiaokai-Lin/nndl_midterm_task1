import lightning as L
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
L.seed_everything(42)

class CubClassifier(L.LightningModule):
    def __init__(self, optimizer_name, lr, pretrained = False):
        super().__init__()
        self.save_hyperparameters()
        self.loss_module = nn.CrossEntropyLoss()
        self.pretrained = pretrained
        if self.pretrained:  
            backbone = resnet18(weights="DEFAULT")
            # self.feature_extractor.eval()
        else:
            backbone = resnet18()
            
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = 200
        self.classifier = nn.Linear(num_filters, num_target_classes)
        
    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        y = self.classifier(representations)
        return y
    
    def configure_optimizers(self):
        # Support Adam or SGD as optimizers.
        # Set different learning rate for backbone and classifier
        classifier_params = list(self.classifier.parameters())
        backbone_params = list(self.feature_extractor.parameters())
        
        if self.hparams.optimizer_name == "Adam":
            grouped_parameters = [
                {"params": backbone_params, "lr": self.hparams.lr / 10},
                {"params": classifier_params, "lr": self.hparams.lr},
            ]
            optimizer = optim.AdamW(grouped_parameters, lr = self.hparams.lr)
        elif self.hparams.optimizer_name == "SGD":
            grouped_parameters = [
                {"params": backbone_params, "lr": self.hparams.lr / 10},
                {"params": classifier_params, "lr": self.hparams.lr},
            ]
            optimizer = optim.SGD(grouped_parameters, lr = self.hparams.lr)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'
        
        # Reduce the learning rate:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss  # Return tensor to call ".backward" on
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = self.loss_module(preds, labels)
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.forward(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)