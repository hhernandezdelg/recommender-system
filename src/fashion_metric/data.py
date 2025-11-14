import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from .config import DATA_PATH, get_default_image_dir


class FashionDataset(Dataset):
    """
    Dataset wrapper that pairs CSV annotations with images stored on disk.
    It loads each row lazily and applies optional transforms.
    """

    def __init__(
        self,
        csv_file: Path,
        img_dir: Path,
        column_class: str = "articleTypeId",
        transform=None,
    ):
        """
        Args:
            csv_file: Path to the CSV file with labels.
            img_dir: Directory with all the images.
            column_class: Column name containing the class / label id.
            transform: Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.targets = list(self.df[column_class].values)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.df.loc[idx, 'imageId']}.jpg")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.targets[idx]


def build_train_transform():
    """
    Builds the image transform used during training. It applies small rotations, 
    color jittering, resizing and normalization so that the model sees slightly varied but consistently scaled inputs.
    """
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=2),
            transforms.RandomRotation(degrees=4),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5
            ),
            transforms.Resize((60, 80)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transform():
    """
    Builds the image transform used for validation and testing. 
    It only resizes and normalizes the images so that evaluation is performed on a stable and comparable representation.
    """
    return transforms.Compose(
        [
            transforms.Resize((60, 80)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_all_datasets(
    img_dir: Path | None = None, data_path: Path | None = None
) -> Dict[str, FashionDataset]:
    """
    Creates all dataset objects used in the experiments from the available CSV files. 
    It connects each CSV file to the shared image directory and applies the appropriate transforms.
    """
    if img_dir is None:
        img_dir = get_default_image_dir()
    if data_path is None:
        data_path = DATA_PATH

    img_dir = Path(img_dir)
    data_path = Path(data_path)

    transform_train = build_train_transform()
    transform_eval = build_eval_transform()

    train_dataset = FashionDataset(data_path / "train.csv", img_dir, transform=transform_eval)
    main_test_dataset = FashionDataset(data_path / "main_test.csv", img_dir, transform=transform_eval)
    new_test_dataset = FashionDataset(data_path / "new_test.csv", img_dir, transform=transform_eval)
    main_support_dataset = FashionDataset(data_path / "main_support.csv", img_dir, transform=transform_eval)
    new_support_dataset = FashionDataset(data_path / "new_support.csv", img_dir, transform=transform_eval)
    merged_test_dataset = FashionDataset(
        data_path / "merged_test.csv",
        img_dir,
        transform=transform_eval,
    )
    merged_support_dataset = FashionDataset(
        data_path / "merged_support.csv",
        img_dir,
        transform=transform_eval,
    )

    # Datasets with categories
    main_test_dataset_cat = FashionDataset(
        data_path / "main_test.csv",
        img_dir,
        column_class="categoryId",
        transform=transform_eval,
    )
    main_support_dataset_cat = FashionDataset(
        data_path / "main_support.csv",
        img_dir,
        column_class="categoryId",
        transform=transform_eval,
    )

    return {
        "train_dataset": train_dataset,
        "main_test_dataset": main_test_dataset,
        "new_test_dataset": new_test_dataset,
        "main_support_dataset": main_support_dataset,
        "new_support_dataset": new_support_dataset,
        "merged_test_dataset": merged_test_dataset,
        "merged_support_dataset": merged_support_dataset,
        "main_test_dataset_cat": main_test_dataset_cat,
        "main_support_dataset_cat": main_support_dataset_cat,
    }


def build_classifier_dataloaders(
    train_idx,
    val_idx,
    img_dir: Path | None = None,
    data_path: Path | None = None,
    batch_size: int = 32,
) -> Tuple[DataLoader, ...]:
    """
    Builds the data loaders used for supervised classifier training. 
    It takes train and validation indices and returns loaders for train validation and test splits with the configured batch size.
    """
    if img_dir is None:
        img_dir = get_default_image_dir()
    if data_path is None:
        data_path = DATA_PATH

    img_dir = Path(img_dir)
    data_path = Path(data_path)

    transform_train = build_train_transform()
    transform_eval = build_eval_transform()

    train_dataset = FashionDataset(data_path / "train.csv", img_dir, transform=transform_train)
    val_dataset = FashionDataset(data_path / "train.csv", img_dir, transform=transform_eval)
    main_test_dataset = FashionDataset(data_path / "main_test.csv", img_dir, transform=transform_eval)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        main_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader
