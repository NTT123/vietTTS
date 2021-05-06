from pathlib import Path
from typing import NamedTuple


class TrainingConfig(NamedTuple):
  end_step: int
  learning_rate: float
  batch_size: int


class FLAGS:
  seq_len = 1024*3
  total_training_steps = 1_000_000
  gru_dim = 896
  embed_dim = 896
  _training_schedule = [
      TrainingConfig(1_000,     1e-6,   8),
      TrainingConfig(2_000,     1e-5,  16),
      TrainingConfig(3_000,     1e-4,  32),
      TrainingConfig(5_000,     3e-4,  64),
      TrainingConfig(50_000,    5e-4, 128),
      TrainingConfig(100_000,   3e-4, 128),
      TrainingConfig(200_000,   1e-4, 128),
      TrainingConfig(300_000,   5e-5, 128),
      TrainingConfig(1_000_000, 1e-5, 128),
  ]

  # training config
  ckpt_dir = Path('assets/reinfo/waveRNN')
  wav_data_dir = Path('train_data')
  sample_rate = 16_000
  variance_loss_scale = 0.1  # regularization term
