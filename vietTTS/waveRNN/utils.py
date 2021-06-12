import pickle
import time

import jax
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from jax._src.numpy.lax_numpy import concatenate
from tabulate import tabulate
from vietTTS.nat.dsp import *

from .config import *
from .model import *


def encode_16bit_coarse_fine(y):
  y = y.astype(jnp.int32) + 2**15
  fine = jnp.bitwise_and(y, int('0b' + '1' * FLAGS.num_fine_bits, 2))
  coarse = jnp.right_shift(y, FLAGS.num_fine_bits)
  y = jnp.stack((coarse, fine), axis=-1).astype(jnp.int32)
  return y


def encode_16bit_mu_law(y, mu=255):
  y = y.astype(jnp.float32) / (2**15)  # [-1, 1]
  mu_y = jnp.sign(y) * jnp.log1p(mu * jnp.abs(y)) / jnp.log1p(mu)
  mu_y = ((mu_y + 1)/2 * mu).astype(jnp.int32)
  return mu_y


def save_checkpoint(step, params, aux, optim_state, path=FLAGS.ckpt_dir):
  params, aux, optim_state = jax.device_get((params, aux, optim_state))
  fn = path / f'checkpoint_{step:08d}.pickle'
  print(f'  > saving checkpoint at {fn}')
  with open(fn, 'wb') as f:
    pickle.dump({'step': step, 'params': params, 'aux': aux, 'optim_state': optim_state}, f)


def load_latest_checkpoint(path=FLAGS.ckpt_dir):
  ckpts = sorted(path.glob('checkpoint_*.pickle'))
  if len(ckpts) == 0:
    print(f'There is no checkpoint in {path}')
    return None
  latest_ckpt = ckpts[-1]
  print(f"  Loading checkpoint from {latest_ckpt}")
  with open(latest_ckpt, 'rb') as f:
    dic = pickle.load(f)
  return dic


def make_new_log_file():
  counter = len(tuple(FLAGS.ckpt_dir.glob('log.*.txt')))
  fn = FLAGS.ckpt_dir / f'log.{counter}.txt'
  print(f'Creating new log file at {fn}')
  return open(fn, 'w', buffering=1)


@hk.transform_with_state
def regenerate_from_signal_(y, sr):
  melfilter = MelFilter(sr, 1024, 80, fmin=FLAGS.fmin, fmax=FLAGS.fmax)
  pad_left = 1024
  pad_right = 1024
  y = y.astype(jnp.float32) / (2**(FLAGS.bits-1))  # rescale
  y = jnp.pad(y, ((0, 0), (pad_left, pad_right)))
  mel = melfilter(y)

  net = WaveRNN(is_training=False)
  c0 = jnp.array([(2**FLAGS.num_coarse_bits)//2])
  f0 = jnp.array([0])
  hx = net.rnn.initial_state(1)
  out = []

  mel = net.upsample(mel)

  def loop(inputs, prev_state):
    mel, rng1, rng2 = inputs
    rng1, rng2 = rng1[0], rng2[0]
    coarse, fine, hx = prev_state
    coarse = net.rnn.c_embed(coarse)
    fine = net.rnn.f_embed(fine)
    x = jnp.concatenate((mel, coarse, fine, coarse), axis=-1)
    (yc, _), _ = net.rnn.step(x, hx)
    clogits = net.rnn.O2(jax.nn.relu(net.rnn.O1(yc)))
    new_coarse_bit = jax.random.categorical(rng1, clogits, axis=-1)
    new_coarse = net.rnn.c_embed(new_coarse_bit)
    x = jnp.concatenate((mel, coarse, fine, new_coarse), axis=-1)
    (_, yf), new_hx = net.rnn.step(x, hx)
    flogits = net.rnn.O4(jax.nn.relu(net.rnn.O3(yf)))
    new_fine_bit = jax.random.categorical(rng2, flogits, axis=-1)

    clogits = jax.nn.softmax(clogits, axis=-1)
    pr = jnp.exp(clogits)
    return (new_coarse_bit, new_fine_bit, pr), (new_coarse_bit, new_fine_bit, new_hx)

  h0 = (c0, f0, hx)
  _, L, _ = mel.shape
  rng1s = jax.random.split(hk.next_rng_key(), L)[None]
  rng2s = jax.random.split(hk.next_rng_key(), L)[None]
  (coarse, fine, pr), _ = hk.dynamic_unroll(loop, (mel, rng1s, rng2s), h0, time_major=False)
  out = (coarse * (2**FLAGS.num_fine_bits) + fine - 2**15).astype(jnp.int16)
  return (out, pr)


regenerate_from_signal = jax.jit(regenerate_from_signal_.apply, static_argnums=[4])


def gen_test_sample(params, aux, rng, test_clip, step=0, sr=16000):
  t1 = time.perf_counter()
  synthesized_clip, pr = regenerate_from_signal(params, aux, rng, test_clip, sr)[0]
  synthesized_clip = jax.device_get(synthesized_clip[0])
  t2 = time.perf_counter()
  delta = t2 - t1
  l = len(synthesized_clip) / sr
  print(f'  take {delta:.3f} seconds to generate {l:.3f} seconds, RTF = {delta / l:.3f}')
  plt.figure(figsize=(20, 6))
  plt.matshow(pr[0, 12000:12100].T, interpolation='nearest', fignum=0, aspect='auto')
  plt.savefig(FLAGS.ckpt_dir / f'predictive_dist_{step}.png')
  plt.close()
  sf.write(str(FLAGS.ckpt_dir/f'generated_clip_{step}.wav'), synthesized_clip, sr)
  sf.write(str(FLAGS.ckpt_dir/f'gt_clip_{step}.wav'), test_clip[0], sr)


def print_flags(dict):
  values = [(k, v) for k, v in dict.items() if not k.startswith('_')]
  print(tabulate(values))
