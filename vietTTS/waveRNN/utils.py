import pickle
import time

import jax
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from tabulate import tabulate
from vietTTS.tacotron.dsp import *

from .config import *
from .model import *


def encode_16bit_mu_law(y, mu=255):
  y = y.astype(jnp.float32) / (2**15)  # [-1, 1]
  mu_y = jnp.sign(y) * jnp.log1p(mu * jnp.abs(y)) / jnp.log1p(mu)
  mu_y = ((mu_y + 1)/2 * 255).astype(jnp.int32)
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


@hk.without_apply_rng
@hk.transform_with_state
def regenerate_from_signal_(y, rng, sr):
  melfilter = MelFilter(sr, 1024, 80)
  pad_left = 1024
  pad_right = 1024
  y = y.astype(jnp.float32) / (2**15)  # rescale
  y = jnp.pad(y, ((0, 0), (pad_left, pad_right)))
  mel = melfilter(y)

  net = WaveRNN(is_training=False)
  x = jnp.array([128])
  hx = net.gru.initial_state(1)
  out = []

  mel = net.upsample(mel)

  def loop(mel, prev_state):
    x, rng, hx = prev_state
    rng1, rng = jax.random.split(rng)
    x = net.input_embed(x) + mel
    x, hx = net.gru(x, hx)
    x = net.o2(jax.nn.relu(net.o1(x)))
    x = jax.nn.log_softmax(x, axis=-1)
    pr = jnp.exp(x)
    v = jnp.linspace(0, 255, 256)[None, None, :]
    mean = jnp.sum(pr * v, axis=-1, keepdims=True)
    variance = jnp.sum(jnp.square(v - mean) * pr, axis=-1, keepdims=True)
    reg = jnp.log(1 + jnp.sqrt(variance))
    x = jax.random.categorical(rng1, x)
    return (x, reg, pr), (x, rng, hx)

  h0 = (x, rng, hx)
  (out, reg, pr), _ = hk.dynamic_unroll(loop, mel, h0, time_major=False)
  return (out, reg, pr)


regenerate_from_signal = jax.jit(regenerate_from_signal_.apply, static_argnums=[4])


def gen_test_sample(params, aux, rng, test_clip, step=0, sr=16000):
  t1 = time.perf_counter()
  synthesized_clip, reg, pr = regenerate_from_signal(params, aux, test_clip, rng, sr)[0]
  synthesized_clip = jax.device_get(synthesized_clip)
  synthesized_clip = librosa.mu_expand(synthesized_clip[0] - 128)
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
