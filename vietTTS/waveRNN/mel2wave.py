import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp
import librosa

from .config import *
from .model import WaveRNN


@hk.without_apply_rng
@hk.transform_with_state
def generate_from_mel_(mel, rng):
  net = WaveRNN(mu_law_bits=FLAGS.mu_law_bits, is_training=False)
  x = jnp.array([128])
  hx = net.gru.initial_state(1)
  out = []

  mel = net.upsample(mel)
  n_elem = 2**FLAGS.mu_law_bits

  def loop(mel, prev_state):
    x, rng, hx = prev_state
    rng1, rng = jax.random.split(rng)
    x = net.input_embed(x) + mel
    x, hx = net.gru(x, hx)
    x = net.o2(jax.nn.relu(net.o1(x)))
    x = jax.nn.log_softmax(x, axis=-1)
    pr = jnp.exp(x)
    v = jnp.linspace(0, n_elem-1, n_elem)[None, None, :]
    mean = jnp.sum(pr * v, axis=-1, keepdims=True)
    variance = jnp.sum(jnp.square(v - mean) * pr, axis=-1, keepdims=True)
    reg = jnp.log(1 + jnp.sqrt(variance))
    x = jax.random.categorical(rng1, x)
    return (x, reg, pr), (x, rng, hx)

  h0 = (x, rng, hx)
  (out, reg, pr), _ = hk.dynamic_unroll(loop, mel, h0, time_major=False)
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
  synthesized_clip, reg, pr = generate_from_mel(params, aux, mel, rng)[0]
  synthesized_clip = jax.device_get(synthesized_clip)
  n_elem = 2**FLAGS.mu_law_bits
  synthesized_clip = librosa.mu_expand(synthesized_clip[0] - n_elem//2, mu=n_elem-1)
  t2 = time.perf_counter()
  delta = t2 - t1
  l = len(synthesized_clip) / FLAGS.sample_rate
  print(f'  take {delta:.3f} seconds to generate {l:.3f} seconds, RTF = {delta / l:.3f}')
  return synthesized_clip
