from pathlib import Path


class FLAGS:
  seq_len = 1024*3
  gru_dim = 1024

  # training config
  batch_size = 128
  learning_rate = 512e-6
  training_steps = 1_000_000

  ckpt_dir = Path('assets/infore/waveRNN')
  wav_data_dir = Path('train_data')
  sample_rate = 16_000
  fmin = 0
  fmax = 8000
  bits = 16
  num_coarse_bits = 10
  num_fine_bits = 6
  assert num_coarse_bits + num_fine_bits == bits
