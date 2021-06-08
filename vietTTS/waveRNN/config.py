from pathlib import Path


class FLAGS:
  seq_len = 1024*3
  gru_dim = 1024

  batch_size = 128
  learning_rate = 512e-6

  # training config
  ckpt_dir = Path('assets/infore/waveRNN')
  wav_data_dir = Path('train_data')
  sample_rate = 16_000
  fmin = 0
  fmax = 8000
  variance_loss_scale = 0.1  # regularization term
  bits = 16
  num_coarse_bits = 10
  num_fine_bits = 6
