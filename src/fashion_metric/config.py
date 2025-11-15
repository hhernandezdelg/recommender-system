import os
from pathlib import Path

from kagglehub import dataset_download

def get_project_root() -> Path:
    """
    Returns the root directory of the project, assuming this file lives in src/fashion_metric/.
    """
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = get_project_root()

# Data directory (CSV files)
DATA_PATH: Path = Path(os.environ.get("FASHION_DATA_DIR", PROJECT_ROOT / "data"))


def get_default_image_dir() -> Path:
    """
    Returns the default image directory.
    """
    # The dataset download is triggered so that the notebooks can import the module even when the data is not yet present
    img_dir_env = os.environ.get("FASHION_IMG_DIR")
    if img_dir_env is not None:
        return Path(img_dir_env)

    # Kagglehub caches the images under the cache directory
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dataset_path = dataset_download("paramaggarwal/fashion-product-images-small")
    return Path(dataset_path) / "images"
