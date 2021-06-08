import random
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import *

all_wav_files = sorted(Path(FLAGS.wav_data_dir).glob('*.wav'))
test_wav_files = all_wav_files[:10]  # ten samples for human evaluation
val_len = (len(all_wav_files) - 10) // 10  # 10% for NLL validation
val_wav_files = all_wav_files[10:val_len+10]
train_wav_files = all_wav_files[val_len+10:]  # 90% for training
assert len(test_wav_files) + len(val_wav_files) + len(train_wav_files) == len(all_wav_files)


data_cache = {}


def make_data_iter_from_file_list(data_files, batch_size, seq_len, mode='train'):
  batch = []
  if mode == 'train':
    while True:
      random.shuffle(data_files)
      for fn in data_files:
        if fn.stem in data_cache:
          short_clip = data_cache[fn.stem]
        else:
          short_clip, _sr = sf.read(fn, dtype='int16')
          data_cache[fn.stem] = short_clip
        L = len(short_clip) - 1 - seq_len
        if L < 0:
          continue
        start = random.randint(0, L)
        end = start + seq_len

        batch.append(short_clip[start:end])
        if len(batch) == batch_size:
          batch = np.stack(batch, axis=0)
          batch = batch.astype(np.float32) * np.random.uniform(0.5, 1.0, size=(batch_size, 1))
          batch = np.rint(batch)
          yield batch
          batch = []
  elif mode == "val":
    y = []
    for fn in data_files:
      if fn.stem in data_cache:
        short_clip = data_cache[fn.stem]
      else:
        short_clip, _sr = sf.read(fn, dtype='int16')
        data_cache[fn.stem] = short_clip
      y.append(short_clip)
    y = np.concatenate(y, axis=0)
    batch = []
    for start_idx in range(0, len(y) - seq_len, seq_len):
      end_idx = start_idx + seq_len
      batch.append(y[start_idx:end_idx])
      if len(batch) == batch_size:
        batch = np.stack(batch, axis=0)
        yield batch
        batch = []
    return  # Stop iterator
  else:
    raise ValueError(f"Not supported data mode: {mode}")


def make_train_data_iter(batch_size):
  return make_data_iter_from_file_list(
      train_wav_files,
      batch_size,
      FLAGS.seq_len,
      mode='train'
  )
