"""
A script to download the InfoRE dataset and textgrid files.
"""
import shutil
from pathlib import Path

import pooch
from pooch import Unzip
from tqdm.cli import tqdm


def download_infore_data():
    """download infore wav files"""
    files = pooch.retrieve(
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k_denoised.zip",
        known_hash="2445527b345fb0b1816ce3c8f09bae419d6bbe251f16d6c74d8dd95ef9fb0737",
        processor=Unzip(),
        progressbar=True,
    )
    data_dir = Path(sorted(files)[0]).parent
    return data_dir


def download_textgrid():
    """download textgrid files"""
    files = pooch.retrieve(
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_tg.zip",
        known_hash="26e4f53025220097ea95dc266657de8d65104b0a17a6ffba778fc016c8dd36d7",
        processor=Unzip(),
        progressbar=True,
    )
    data_dir = Path(sorted(files)[0]).parent
    return data_dir


DATA_ROOT = Path("./train_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
wav_dir = download_infore_data()
tg_dir = download_textgrid()

for path in tqdm(tg_dir.glob("*.TextGrid")):
    wav_name = path.with_suffix(".wav").name
    wav_src = wav_dir / wav_name
    shutil.copy(path, DATA_ROOT)
    shutil.copy(wav_src, DATA_ROOT)
