import requests
from pathlib import Path
from typing import Tuple

import torch
from tqdm import tqdm

from .vae import AutoencoderKL
from .text_encoder import FrozenCLIPEmbedder
from .unet import UNetModel


def download_file(url: str, save_dir: Path, filename: str) -> Path:
    """
    Download a file from a URL to a given directory.

    Args:
        url: The URL of the file to download.
        save_dir: The directory to save the file to.
        filename: The name of the file to download.

    Returns:
        Path: The path to the downloaded file.
    """
    # Download the file if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    if path.exists():
        return path

    # Stream download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True,
            desc=f"Downloading {filename}"
        ) as pbar:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return path


def download_models() -> Tuple[Path, Path, Path]:
    """
    Download the Stable Diffusion 1.5 models from Hugging Face.

    Returns:
        Tuple[Path, Path, Path]: The paths to the downloaded models.
    """
    home_dir = Path.home()
    cache_models_dir = Path(home_dir) / ".cache" / "edufusion" / "models" / "stable-diffusion-v1-5"
    cache_models_dir.mkdir(parents=True, exist_ok=True)

    text_encoder_path = download_file(
        "https://huggingface.co/andraniksargsyan/stable-diffusion-v1-5/resolve/main/text-encoder-sd-v1-5-fp16.pt?download=true",
        cache_models_dir,
        "text-encoder-sd-v1-5-fp16.pt"
    )
    vae_path = download_file(
        "https://huggingface.co/andraniksargsyan/stable-diffusion-v1-5/resolve/main/vae-sd-v1-5-fp16.pt?download=true",
        cache_models_dir,
        "vae-sd-v1-5-fp16.pt"
    )
    unet_path = download_file(
        "https://huggingface.co/andraniksargsyan/stable-diffusion-v1-5/resolve/main/unet-sd-v1-5-fp16.pt?download=true",
        cache_models_dir,
        "unet-sd-v1-5-fp16.pt"
    )

    return vae_path, text_encoder_path, unet_path


def load_models(device: str = "cuda") -> Tuple[AutoencoderKL, FrozenCLIPEmbedder, UNetModel]:
    """
    Load the Stable Diffusion 1.5 models from the cache directory.

    Args:
        device: The device to load the models to.

    Returns:
        Tuple[AutoencoderKL, FrozenCLIPEmbedder, UNetModel]: The loaded models.
    """
    vae_path, text_encoder_path, unet_path = download_models()

    text_encoder = FrozenCLIPEmbedder().to(device=device).half()
    text_encoder.load_state_dict(torch.load(text_encoder_path, weights_only=True))

    vae = AutoencoderKL().to(device=device).half()
    vae.load_state_dict(torch.load(vae_path, weights_only=True))

    unet = UNetModel().to(device=device).half()
    unet.load_state_dict(torch.load(unet_path, weights_only=True))

    return vae, text_encoder, unet
