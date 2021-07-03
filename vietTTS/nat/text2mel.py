import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import FLAGS
from .data_loader import load_phonemes_set_from_lexicon_file
from .model import NATNet


def load_lexicon(fn):
  lines = open(fn, 'r').readlines()
  lines = [l.lower().strip().split('\t') for l in lines]
  return dict(lines)


def text2tokens(text, lexicon_fn):
  phonemes = load_phonemes_set_from_lexicon_file(lexicon_fn)
  lexicon = load_lexicon(lexicon_fn)

  words = text.strip().lower().split()
  tokens = [FLAGS.sil_index]
  for word in words:
    if word in FLAGS.special_phonemes:
      tokens.append(phonemes.index(word))
    elif word in lexicon:
      p = lexicon[word]
      p = p.split()
      p = [phonemes.index(pp) for pp in p]
      tokens.extend(p)
      tokens.append(FLAGS.word_end_index)
    else:
      w = word
      while len(w) > 0:
        for l in [3, 2, 1]:
          if w[0:l] in phonemes:
            tokens.append(phonemes.index(w[0:l]))
            w = w[l:]
            break
        else:
          w = w[1:]
      tokens.append(FLAGS.word_end_index)
  tokens.append(FLAGS.sp_index)  # silence
  return tokens


def predict_mel(tokens, silence_duration, speaker):
  ckpt_fn = FLAGS.ckpt_dir / 'nat_ckpt_latest.pickle'
  with open(ckpt_fn, 'rb') as f:
    dic = pickle.load(f)
    last_step, params, aux, rng, optim_state = dic['step'], dic['params'], dic['aux'], dic['rng'], dic['optim_state']

  @hk.transform_with_state
  def forward(tokens, silence_duration, speaker):
    net = NATNet(is_training=False)
    return net.inference(tokens, silence_duration, speaker)

  predict_fn = forward.apply
  tokens = np.array(tokens, dtype=np.int32)[None, :]
  return predict_fn(params, aux, rng, tokens, silence_duration, speaker)[0]


def text2mel(text: str, lexicon_fn=FLAGS.data_dir / 'lexicon.txt', silence_duration: float = -1., speaker: int = 0):
  tokens = text2tokens(text, lexicon_fn)
  mels = predict_mel(tokens, silence_duration, speaker)
  return mels


if __name__ == '__main__':
  from argparse import ArgumentParser
  from pathlib import Path
  parser = ArgumentParser()
  parser.add_argument('--text', type=str, required=True)
  parser.add_argument('--speaker', type=int, default=0)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  mel = text2mel(args.text, speaker=args.speaker)
  plt.figure(figsize=(10, 5))
  plt.imshow(mel[0].T, origin='lower', aspect='auto')
  plt.savefig(str(args.output))
  plt.close()
  mel = jax.device_get(mel)
  mel.tofile('clip.mel')
