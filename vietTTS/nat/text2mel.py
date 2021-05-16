import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import FLAGS, DurationInput
from .data_loader import load_phonemes_set_from_lexicon_file
from .model import AcousticModel, DurationModel


def load_lexicon(fn):
  lines = open(fn, 'r').readlines()
  lines = [l.lower().strip().split('\t') for l in lines]
  return dict(lines)


def predict_duration(tokens):
  forward_fn = jax.jit(hk.transform_with_state(
      lambda x: DurationModel(is_training=False)(x)
  ).apply)
  with open(FLAGS.ckpt_dir / 'duration_ckpt_latest.pickle', 'rb') as f:
    dic = pickle.load(f)
  x = DurationInput(
      np.array(tokens, dtype=np.int32)[None, :],
      np.array([len(tokens)], dtype=np.int32),
      None
  )
  return forward_fn(dic['params'], dic['aux'], dic['rng'], x)[0]


def text2tokens(text, lexicon_fn):
  phonemes = load_phonemes_set_from_lexicon_file(lexicon_fn)
  lexicon = load_lexicon(lexicon_fn)

  words = text.strip().lower().split()
  tokens = [0]
  for word in words:
    if word in phonemes:
      tokens.append(phonemes.index(word))
    elif word in lexicon:
      p = lexicon[word]
      p = p.split()
      p = [phonemes.index(pp) for pp in p]
      tokens.extend(p)
    else:
      for p in word:
        if p in phonemes:
          tokens.append(phonemes.index(p))
  tokens.append(0)  # silence
  return tokens


def predict_mel(tokens, durations):
  ckpt_fn = FLAGS.ckpt_dir / 'acoustic_ckpt_latest.pickle'
  with open(ckpt_fn, 'rb') as f:
    dic = pickle.load(f)
    last_step, params, aux, rng, optim_state = dic['step'], dic['params'], dic['aux'], dic['rng'], dic['optim_state']

  @hk.transform_with_state
  def forward(tokens, durations, n_frames):
    net = AcousticModel(is_training=False)
    return net.inference(tokens, durations, n_frames)

  durations = durations / 10 * FLAGS.sample_rate / (FLAGS.n_fft//4)
  n_frames = int(jnp.sum(durations).item())
  predict_fn = jax.jit(forward.apply, static_argnums=[5])
  tokens = np.array(tokens, dtype=np.int32)[None, :]
  return predict_fn(params, aux, rng, tokens, durations, n_frames)[0]


def text2mel(text: str, lexicon_fn=FLAGS.data_dir / 'lexicon.txt', silence_duration: float = -1.):
  tokens = text2tokens(text, lexicon_fn)
  durations = predict_duration(tokens)
  durations = jnp.where(
      np.array(tokens)[None, :] <= 2,
      jnp.clip(durations, a_min=silence_duration * 10, a_max=10),
      durations
  )
  mels = predict_mel(tokens, durations)
  end_silence = durations[0, -1].item() / 10
  silence_frame = int(end_silence * FLAGS.sample_rate / (FLAGS.n_fft // 4))
  return mels[:, :-silence_frame]


if __name__ == '__main__':
  from argparse import ArgumentParser
  from pathlib import Path
  parser = ArgumentParser()
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  mel = text2mel(args.text)
  plt.figure(figsize=(10, 5))
  plt.imshow(mel[0].T, origin='lower', aspect='auto')
  plt.savefig(str(args.output))
  plt.close()
  mel = jax.device_get(mel)
  mel.tofile('clip.mel')
