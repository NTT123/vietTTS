import random
from pathlib import Path

import numpy as np
import textgrid
from scipy.io import wavfile
from tqdm.auto import tqdm

from .config import FLAGS, AcousticInput


def load_phonemes_set_from_lexicon_file(fn: Path):
  S = set()
  for line in open(fn, 'r').readlines():
    word, phonemes = line.strip().lower().split('\t')
    phonemes = phonemes.split()
    S.update(phonemes)

  S = FLAGS.special_phonemes + sorted(list(S))
  return S


def pad_seq(s, maxlen, value=0, check_length=True):
  if check_length:
    assert maxlen >= len(s)
  else:
    if len(s) > maxlen:
      s = s[:maxlen]
  return tuple(s) + (value,) * (maxlen - len(s))


def is_in_word(phone, word):
  def time_in_word(time, word):
    return (word.minTime - 1e-3) < time and (word.maxTime + 1e-3) > time
  return time_in_word(phone.minTime, word) and time_in_word(phone.maxTime, word)


def load_textgrid(fn: Path):
  tg = textgrid.TextGrid.fromFile(str(fn.resolve()))
  data = []
  words = list(tg[0])
  widx = 0
  assert tg[1][0].minTime == 0, "The first phoneme has to start at time 0"
  for p in tg[1]:
    if not p in words[widx]:
      widx = widx + 1
      if len(words[widx-1].mark) > 0:
        data.append((FLAGS.special_phonemes[FLAGS.word_end_index], 0.0))
      if widx >= len(words):
        break
      assert p in words[widx], 'mismatched word vs phoneme'
    data.append((p.mark.strip().lower(), p.duration()))
  return data


def frame_idx_encode(durations, forward=True):
  out = []
  end_frame_idx = [0]
  t = 0.0
  for d in durations:
    t = t + d
    end_frame_idx.append(int(t * FLAGS.sample_rate / (FLAGS.n_fft // 4)))
    num_frames = end_frame_idx[-1] - end_frame_idx[-2]
    if forward:
      out.extend(range(num_frames))
    else:
      out.extend(reversed(list(range(num_frames))))
  out.append(0)
  return out


def load_textgrid_wav(data_dir: Path, token_seq_len: int, batch_size, pad_wav_len, mode: str):
  tg_files = sorted(data_dir.glob('*/*.TextGrid'))
  random.Random(42).shuffle(tg_files)
  L = len(tg_files) * 95 // 100
  assert mode in ['train', 'val', 'gta']
  phonemes = load_phonemes_set_from_lexicon_file(data_dir / 'lexicon.txt')
  all_speakers = sorted(set([fn.parent.stem for fn in tg_files]))
  if mode == 'gta':
    tg_files = tg_files  # all files
  elif mode == 'train':
    tg_files = tg_files[:L]
  elif mode == 'val':
    tg_files = tg_files[L:]

  data = []
  for fn in tqdm(tg_files, desc='load data', disable=(mode == 'val')):
    ps, ds = zip(*load_textgrid(fn))
    ps = [phonemes.index(p) for p in ps]
    duration_length = len(ps)
    ps = pad_seq(ps, token_seq_len, 0)
    ds = pad_seq(ds, token_seq_len, 0)
    fs1 = frame_idx_encode(ds, forward=True)
    fs1 = pad_seq(fs1, pad_wav_len // (FLAGS.n_fft // 4), 0, False)
    fs2 = frame_idx_encode(ds, forward=False)
    fs2 = pad_seq(fs2, pad_wav_len // (FLAGS.n_fft // 4), 0, False)
    wav_file = fn.parent / f'{fn.stem}.wav'
    sr, y = wavfile.read(wav_file)
    y = np.copy(y)
    start_time = 0
    for i, (phone_idx, duration) in enumerate(zip(ps, ds)):
      l = int(start_time * sr)
      end_time = start_time + duration
      r = int(end_time * sr)
      if i == len(ps) - 1:
        r = len(y)
      if phone_idx < len(FLAGS.special_phonemes):
        y[l: r] = 0
      start_time = end_time

    if len(y) > pad_wav_len:
      y = y[:pad_wav_len]
    else:
      wav_length = len(y)
      y = np.pad(y, (0, pad_wav_len - len(y)))
    spk = all_speakers.index(fn.parent.stem)
    data.append((fn.stem, ps, ds, duration_length, y, wav_length, fs1, fs2, spk))

  batch = []
  while True:
    random.shuffle(data)
    for idx, e in enumerate(data):
      batch.append(e)
      if len(batch) == batch_size or (mode == 'gta' and idx == len(data) - 1):
        names, ps, ds, lengths, wavs, wav_lengths, fs1, fs2, spks = zip(*batch)
        ps = np.array(ps, dtype=np.int32)
        fs1 = np.array(fs1, dtype=np.int32)
        fs2 = np.array(fs2, dtype=np.int32)
        ds = np.array(ds, dtype=np.float32)
        lengths = np.array(lengths, dtype=np.int32)
        spks = np.array(spks, dtype=np.int32)
        wavs = np.array(wavs)
        wav_lengths = np.array(wav_lengths, dtype=np.int32)
        if mode == 'gta':
          yield names, AcousticInput(ps, lengths, ds, wavs, wav_lengths, None, fs1, fs2, spks)
        else:
          yield AcousticInput(ps, lengths, ds, wavs, wav_lengths, None, fs1, fs2, spks)
        batch = []
    if mode == 'gta':
      assert len(batch) == 0
      break
