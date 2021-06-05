from pathlib import Path
from typing import NamedTuple


class TrainingConfig(NamedTuple):
  end_step: int
  learning_rate: float
  batch_size: int


class FLAGS:
  seq_len = 1024*3
  gru_dim = 1024
  _training_schedule = [
      TrainingConfig(1_000,     1e-6,   8),
      TrainingConfig(2_000,     1e-5,  16),
      TrainingConfig(3_000,     1e-4,  32),
      TrainingConfig(5_000,     3e-4,  64),
      TrainingConfig(100_000,   5e-4, 128),
      TrainingConfig(200_000,   3e-4, 128),
      TrainingConfig(300_000,   1e-4, 128),
      TrainingConfig(500_000,   5e-5, 128),
      TrainingConfig(1_000_000, 1e-5, 128),
      TrainingConfig(2_000_000, 5e-6, 128),
      TrainingConfig(3_000_000, 1e-6, 128),
  ]

  # training config
  ckpt_dir = Path('assets/infore/waveRNN')
  wav_data_dir = Path('train_data')
  sample_rate = 16_000
  fmin = 0
  fmax = 8000
  variance_loss_scale = 0.1  # regularization term
  bits = 16
  mu_law_bits = 8
