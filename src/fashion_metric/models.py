from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class CNET_HHD0_V44(nn.Module):
    """
    Convolutional neural network used to extract features from fashion images and perform multi class classification. 
    It stacks several convolutional blocks with pooling and dropout to build a compact representation from low resolution inputs. 
    The classifier head maps the flattened features to the requested number of classes.
    """

    def __init__(
        self,
        num_classes: int = 315,
        input_shape=(3, 60, 80),
        class_weights=None,
        in_ch: int = 256,
        drop_1: float = 0.0,
        drop_2: float = 0.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.drop_1 = drop_1
        self.drop_2 = drop_2
        self.input_shape = input_shape

        self.class_weights = None

        # Feature extractor blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(3, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(drop_1 + 0.05),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(drop_1 + 0.05),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 4),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(drop_1 + 0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_ch * 4, in_ch * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 8),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(drop_1 + 0.15),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_ch * 8, in_ch * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 16),
            nn.GELU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(drop_1 + 0.2),
        )

        # Compute flattened size
        with torch.no_grad():
            x_dummy = torch.zeros(1, *input_shape)
            x_dummy = self.block1(x_dummy)
            x_dummy = self.block2(x_dummy)
            x_dummy = self.block3(x_dummy)
            x_dummy = self.block4(x_dummy)
            x_dummy = self.block5(x_dummy)
            self._flattened_size = x_dummy.view(1, -1).size(1)

        # Classifier block
        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_size, in_ch * 16),
            nn.LayerNorm(in_ch * 16),
            nn.ELU(),
            nn.Dropout(drop_2),
            nn.Linear(in_ch * 16, num_classes),
            nn.BatchNorm1d(num_classes),
        )

    def forward(self, x, return_embeddings: bool = False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        feats = x
        x = self.classifier(x)
        probs = F.softmax(x, dim=1)

        if return_embeddings:
            x = F.normalize(feats, p=2, dim=1)
            return x
        return probs

    def get_embedding(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        return torch.nn.functional.normalize(x, p=2, dim=1)


def build_HHD_model(
    num_classes: int = 39,
    in_ch: int = 256,
    drop_1: float = 0.1,
    drop_2: float = 0.1,
) -> CNET_HHD0_V44:
    """
    Builds the main convolutional neural network model used for fashion image classification. 
    The arguments control the number of output classes and some capacity and regularization settings.
    """
    model = CNET_HHD0_V44(
        num_classes=num_classes,
        in_ch=in_ch,
        drop_1=drop_1,
        drop_2=drop_2,
    )
    return model


def configure_finetuning(model: nn.Module) -> nn.Module:
    """
    Marks all parameters of the model as trainable so that they are updated during optimization. 
    This is useful when we want to fine tune the full network.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model
