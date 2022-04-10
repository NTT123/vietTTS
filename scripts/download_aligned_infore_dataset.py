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
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k.zip",
        known_hash="0c9b2fd6962fd6706fa9673f94a9f1ba534edf34691247ae2be0ff490870ccd7",
        processor=Unzip(),
        progressbar=True,
    )
    data_dir = Path(sorted(files)[0]).parent
    return data_dir


def download_textgrid():
    """download textgrid files"""
    files = pooch.retrieve(
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_tg.zip",
        known_hash="b5e18152c1b3957d750218bc96bf11d1580c0b056662a29b34c0cccab7376e86",
        processor=Unzip(),
        progressbar=True,
    )
    data_dir = Path(sorted(files)[0]).parent
    return data_dir


DATA_ROOT = Path("./train_data")  # modify this
DATA_ROOT.mkdir(parents=True, exist_ok=True)
wav_dir = download_infore_data()
tg_dir = download_textgrid()

for path in tqdm(tg_dir.glob("*.TextGrid")):
    wav_name = path.with_suffix(".wav").name
    wav_src = wav_dir / wav_name
    shutil.copy(path, DATA_ROOT)
    shutil.copy(wav_src, DATA_ROOT)
