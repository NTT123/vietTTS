import os
import pickle

import jax
import matplotlib.pyplot as plt

from .config import *


def plot_attn(aux, L, LL, step=-1):
  plt.figure(figsize=(9, 3))
  plt.imshow(
      jax.device_get(aux['tacotron']['attn'])[:L, :LL],
      interpolation='nearest',
      aspect='auto',
  )
  plt.savefig(FLAGS.ckpt_dir / f'attn_{step}.png')
  plt.close()


def save_ckpt(step, params, aux, optim_state, rng):
  fn = FLAGS.ckpt_dir / 'latest_state.pickle'
  print(f'  > saving checkpoint at {fn}')
  with open(fn, 'wb') as f:
    pickle.dump(
        {'step': step,
         'params': params,
         'aux': aux,
         'optim_state': optim_state,
         'rng': rng
         }, f)


def load_latest_ckpt():
  if os.path.exists(FLAGS.ckpt_dir / 'latest_state.pickle'):
    with open(FLAGS.ckpt_dir / 'latest_state.pickle', 'rb') as f:
      dic = pickle.load(f)
      return dic['step'], dic['params'], dic['aux'], dic['optim_state'], dic['rng']
  else:
    return None


def make_new_log_file():
  counter = len(tuple(FLAGS.ckpt_dir.glob('log.*.txt')))
  fn = FLAGS.ckpt_dir / f'log.{counter}.txt'
  print(f'Creating new log file at {fn}')
  return open(fn, 'w', buffering=1)


def print_flags(dict):
  values = [(k, v) for k, v in dict.items() if not k.startswith('_')]
  print(tabulate(values))
