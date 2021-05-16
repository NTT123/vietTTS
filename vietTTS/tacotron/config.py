from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

from einops import rearrange
from jax.numpy import ndarray
from tabulate import tabulate


class TacotronInput(NamedTuple):
  text: ndarray
  text_len: ndarray
  mel: ndarray


class TacotronOutput(NamedTuple):
  stop_token: ndarray
  mel: ndarray
  mel_residual: ndarray


def print_dict(d):
  l = [(k, v) for (k, v) in d.__dict__.items() if not k.startswith('_')]
  headers, values = zip(*l)
  print(tabulate(l))


def make_alphabet():
  import unicodedata
  _pad = '_'
  _punctuation = '!\'(),.:;? '
  _special = '-#'  # `#` for a [silent] duration of 0.05s
  alphabet = unicodedata.normalize(
      'NFKC',
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
      "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằ"
      "ẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữ"
      "ỰựỲỳỴỵỶỷỸỹ"
  )

  symbols = [_pad] + list(_punctuation) + list(_special) + list(alphabet)
  return symbols


class TrainConfig(NamedTuple):
  end_step: int
  reduce_factor: int
  learning_rate: float
  mel_dropout: float


class FLAGS(Namespace):
  # trainer
  batch_size: int = 32
  training_steps: int = 1_000_000
  logging_freq: int = 100

  max_reduce_factor = 32
  _training_schedule = [
      TrainConfig(5_000,   32, 1e-4, 0.5),
      TrainConfig(6_000,   16, 1e-6, 0.5),
      TrainConfig(10_000,  16, 1e-4, 0.5),
      TrainConfig(11_000,   8, 1e-6, 0.5),
      TrainConfig(20_000,   8, 1e-4, 0.5),
      TrainConfig(21_000,   4, 1e-6, 0.5),
      TrainConfig(40_000,   4, 1e-4, 0.5),
      TrainConfig(41_000,   2, 1e-6, 0.5),
      TrainConfig(80_000,   2, 1e-4, 0.5),
      TrainConfig(81_000,   1, 1e-6, 0.5),
      TrainConfig(100_000,  1, 1e-4, 0.5),
      TrainConfig(110_000,  1, 1e-4, 0.4),
      TrainConfig(120_000,  1, 1e-4, 0.3),
      TrainConfig(130_000,  1, 1e-4, 0.2),
      TrainConfig(140_000,  1, 5e-5, 0.1),
      TrainConfig(150_000,  1, 3e-5, 0.0),
      TrainConfig(200_000,  1, 1e-5, 0.0),
  ]

  ckpt_dir = Path('assets/infore/tacotron')
  data_dir = Path('/tmp/infore/raw/pp')

  # dsp
  sample_rate = 16_000
  mel_dim = 80
  pad_mel_len = 800
  pad_text_len = 200
  n_fft = 1024
  fmin = 0.0
  fmax = 8000

  # encoder
  _alphabet = make_alphabet()
  alphabet_size = 512
  text_embed_dim: int = 512
  text_lstm_dim: int = 512

  # decoder
  dec_lstm_dim = 1024

  # attention
  attn_lstm_dim = 1024

  # postnet
  postnet_dim = 512
