import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp
import librosa

from .config import *
from .model import WaveRNN


@hk.transform_with_state
def generate_from_mel_(mel):
  net = WaveRNN(mu_law_bits=FLAGS.mu_law_bits, is_training=False)
  n_elem = 2**FLAGS.mu_law_bits
  c0 = jnp.array([127])
  f0 = jnp.array([0])
  hx = net.gru.initial_state(1)
  out = []

  mel = net.upsample(mel)
  _, L, D = mel.shape

  def loop(inputs, prev_state):
    mel, rng1, rng2 = inputs
    rng1, rng2 = rng1[0], rng2[0]
    coarse, fine, hx = prev_state
    coarse = net.rnn.c_embed(coarse)
    fine = net.rnn.f_embed(fine)
    x = jnp.concatenate((mel, coarse, fine, coarse), axis=-1)
    (yc, _), _ = net.rnn.step(x, hx)
    clogits = net.rnn.O2(jax.nn.relu(net.rnn.O1(yc)))
    new_coarse_8bit = jax.random.categorical(rng1, clogits, axis=-1)
    new_coarse = net.rnn.c_embed(new_coarse_8bit)
    x = jnp.concatenate((mel, coarse, fine, new_coarse), axis=-1)
    (_, yf), new_hx = net.rnn.step(x, hx)
    flogits = net.rnn.O4(jax.nn.relu(net.rnn.O3(yf)))
    new_fine_8bit = jax.random.categorical(rng2, flogits, axis=-1)

    clogits = jax.nn.softmax(clogits, axis=-1)
    pr = jnp.exp(clogits)
    v = jnp.linspace(0, n_elem-1, n_elem)[None, :]
    mean = jnp.sum(pr * v, axis=-1, keepdims=True)
    variance = jnp.sum(jnp.square(v - mean) * pr, axis=-1, keepdims=True)
    reg = jnp.log(1 + jnp.sqrt(variance))
    return (new_coarse_8bit, new_fine_8bit, reg, pr), (new_coarse_8bit, new_fine_8bit, rng, new_hx)

  rng1s = jax.random.split(hk.next_rng_key(), L)[None]
  rng2s = jax.random.split(hk.next_rng_key(), L)[None]
  h0 = (c0, f0, hx)
  (coarse, fine, reg, pr), _ = hk.dynamic_unroll(loop, (mel, rng1s, rng2s), h0, time_major=False)
  out = (coarse * 256 + fine - 2**15).astype(jnp.int16)
  return (out, reg, pr)


generate_from_mel = jax.jit(generate_from_mel_.apply)


def load_latest_checkpoint(path):
  ckpts = sorted(path.glob('checkpoint_*.pickle'))
  if len(ckpts) == 0:
    print(f'There is no checkpoint in {path}')
    return None
  latest_ckpt = ckpts[-1]
  print(f"Loading checkpoint from {latest_ckpt}")
  with open(latest_ckpt, 'rb') as f:
    dic = pickle.load(f)
  return dic


def mel2wave(mel):

  dic = load_latest_checkpoint(FLAGS.ckpt_dir)

  training_step = dic['step']
  params = dic['params']
  aux = dic['aux']
  optim_state = dic['optim_state']

  t1 = time.perf_counter()
  rng = jax.random.PRNGKey(42)
  synthesized_clip, reg, pr = generate_from_mel(params, aux, rng, mel)[0]
  synthesized_clip = jax.device_get(synthesized_clip)
  n_elem = 2**FLAGS.mu_law_bits
  synthesized_clip = librosa.mu_expand(synthesized_clip[0] - n_elem//2, mu=n_elem-1)
  t2 = time.perf_counter()
  delta = t2 - t1
  l = len(synthesized_clip) / FLAGS.sample_rate
  print(f'  take {delta:.3f} seconds to generate {l:.3f} seconds, RTF = {delta / l:.3f}')
  return synthesized_clip
