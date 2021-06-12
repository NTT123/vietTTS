import time
from typing import Deque

import optax
import soundfile as sf
from vietTTS.nat.dsp import MelFilter

from .config import *
from .data_loader import *
from .model import *
from .utils import *


@hk.without_apply_rng
@hk.transform_with_state
def net(x, m): return WaveRNN()(x, m)


def cross_entropy_logits_sparse(logits, labels):
  labeled_logits = jnp.take_along_axis(logits, labels[..., None], -1).squeeze(-1)
  return jax.nn.logsumexp(logits, axis=-1) - labeled_logits


def loss_fn(params, aux, batch, sr=16000):
  melfilter = MelFilter(sr, 1024, 80, fmin=FLAGS.fmin, fmax=FLAGS.fmax)
  y = batch
  mu = encode_16bit_coarse_fine(y)
  y = y.astype(jnp.float32) / (2**15)
  mel = melfilter(y)
  pad = 1024
  mu = mu[:, (pad-1):-pad]
  muinputs = mu[:, :-1]
  mutargets = mu[:, 1:]
  (clogpr, flogpr), aux = net.apply(params, aux, muinputs, mel)
  cllh = cross_entropy_logits_sparse(clogpr, mutargets[..., 0])
  fllh = cross_entropy_logits_sparse(flogpr, mutargets[..., 1])
  loss = jnp.mean(cllh + fllh)
  return loss, aux


def make_optim():
  return optax.chain(
      optax.clip_by_global_norm(1.),
      optax.adam(
          optax.exponential_decay(FLAGS.learning_rate, 100_000, 0.5, False, 1e-6)
      )
  )


def train():
  data_iter = make_train_data_iter(FLAGS.batch_size)
  # generate initial states
  next(data_iter).shape, next(data_iter).dtype
  rng = jax.random.PRNGKey(42)
  y = next(data_iter)
  mu = encode_16bit_coarse_fine(y)
  sr = FLAGS.sample_rate
  melfilter = MelFilter(sr, 1024, 80)
  mel = melfilter(y.astype(jnp.float32)/(2**15))
  pad = 1024
  mu = mu[:, pad-1:-pad]
  muinputs = mu[:, :-1]
  params, aux = net.init(rng, muinputs, mel)
  optimizer = make_optim()
  optim_state = optimizer.init(params)
  training_step = -1

  @jax.jit
  def update(params, aux, optim_state, batch):
    (loss, new_aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, aux, batch)
    updates, new_optim_state = optimizer.update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_aux, new_optim_state

  logfile = make_new_log_file()
  dic = load_latest_checkpoint()
  if dic is not None:
    training_step = dic['step']
    params = dic['params']
    aux = dic['aux']
    optim_state = dic['optim_state']

  test_clip, _sr = sf.read(test_wav_files[0], dtype='int16')
  losses = Deque(maxlen=100)
  l1s = Deque(maxlen=100)
  l2s = Deque(maxlen=100)
  start = time.perf_counter()
  total_training_steps = FLAGS.training_steps
  for step in range(training_step + 1, 1 + total_training_steps):
    training_step += 1
    loss, params, aux, optim_state = update(params, aux, optim_state, next(data_iter))
    losses.append(loss)

    if step % 100 == 0:
      end = time.perf_counter()
      speed = 100 / (end - start)
      start = end
      loss = sum(losses, 0.0).item() / len(losses)
      msg = (f'  {step:06d} | train loss {loss:.3f} | {speed:.3f} it/s ')
      print(msg)
      logfile.write(msg + '\n')

    if step % 1000 == 0:
      save_checkpoint(training_step, params, aux, optim_state)

    # generate test samples
    if step % 1000 == 0:
      gen_test_sample(params, aux, rng, test_clip=test_clip[None, :], step=step, sr=sr)


if __name__ == '__main__':
  if not FLAGS.ckpt_dir.exists():
    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
  print_flags(FLAGS.__dict__)
  train()
