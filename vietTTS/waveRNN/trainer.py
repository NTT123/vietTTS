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
def net(x, m): return WaveRNN(mu_law_bits=FLAGS.mu_law_bits)(x, m)


def loss_fn(params, aux, batch, sr=16000):
  melfilter = MelFilter(sr, 1024, 80, fmin=FLAGS.fmin, fmax=FLAGS.fmax)
  y = batch
  n_elem = 2**FLAGS.mu_law_bits
  mu = encode_16bit_mu_law(y, mu=n_elem - 1)
  y = y.astype(jnp.float32) / (2**15)
  mel = melfilter(y)
  pad = 1024
  mu = mu[:, (pad-1):-pad]
  muinputs = mu[:, :-1]
  mutargets = mu[:, 1:]
  logpr, aux = net.apply(params, aux, muinputs, mel)
  pr = jnp.exp(logpr)
  v = jnp.linspace(0, n_elem-1, n_elem)[None, None, :]
  mean = jnp.sum(pr * v, axis=-1, keepdims=True)
  variance = jnp.sum(jnp.square(v - mean) * pr, axis=-1, keepdims=True)
  reg = jnp.log(1 + jnp.sqrt(variance))
  targets = jax.nn.one_hot(mutargets, num_classes=n_elem)
  llh = jnp.sum(targets * logpr, axis=-1)
  l1 = -jnp.mean(llh)
  l2 = FLAGS.variance_loss_scale * jnp.mean(reg)
  return l1 + l2, (l1, l2, aux)


def make_optim(lr):
  return optax.chain(
      optax.clip_by_global_norm(1.),
      optax.scale_by_adam(),
      optax.scale(-lr)
  )


def train():
  data_iter = make_train_data_iter(FLAGS._training_schedule[0].batch_size)
  # generate initial states
  next(data_iter).shape, next(data_iter).dtype
  rng = jax.random.PRNGKey(42)
  y = next(data_iter)
  n_elem = 2**FLAGS.mu_law_bits
  mu = encode_16bit_mu_law(y, mu=n_elem - 1)
  sr = FLAGS.sample_rate
  melfilter = MelFilter(sr, 1024, 80)
  mel = melfilter(y.astype(jnp.float32)/(2**15))
  pad = 1024
  mu = mu[:, pad-1:-pad]
  muinputs = mu[:, :-1]
  params, aux = net.init(rng, muinputs, mel)
  optimizer = make_optim(FLAGS._training_schedule[0].learning_rate)
  optim_state = optimizer.init(params)
  training_step = -1

  @jax.jit
  def update(params, aux, optim_state, batch, learning_rate):
    (loss, (l1, l2, new_aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, aux, batch)
    optimizer = make_optim(learning_rate)
    updates, new_optim_state = optimizer.update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    return (loss, l1, l2), new_params, new_aux, new_optim_state

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
  training_config = FLAGS._training_schedule[0]
  total_training_steps = FLAGS._training_schedule[-1].end_step
  for step in range(training_step + 1, 1 + total_training_steps):
    training_step += 1
    # update train config
    if step % 1000 == 0:
      for config in FLAGS._training_schedule:
        if step < config.end_step:
          training_config = config
          data_iter = make_train_data_iter(training_config.batch_size)
          break

    (loss, l1, l2), params, aux, optim_state = update(
        params,
        aux,
        optim_state,
        next(data_iter),
        training_config.learning_rate
    )
    l1s.append(l1)
    l2s.append(l2)

    if step % 100 == 0:
      end = time.perf_counter()
      speed = 100 / (end - start)
      start = end
      l1 = sum(l1s, 0.0).item() / len(l1s)
      l2 = sum(l2s, 0.0).item() / len(l2s)
      msg = (f'  {step:06d} | train loss {l1:.3f} | reg loss {l2:.3f} | {speed:.3f} it/s | '
             f'LR {training_config.learning_rate:.3e} | batch size {training_config.batch_size} ')
      print(msg)
      logfile.write(msg + '\n')

    if step % 10000 == 0:
      save_checkpoint(
          training_step,
          params,
          aux,
          optim_state,
      )

    # generate test samples
    if step % 10000 == 0:
      gen_test_sample(params, aux, rng, test_clip=test_clip[None, :], step=step, sr=sr)


if __name__ == '__main__':
  if not FLAGS.ckpt_dir.exists():
    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
  print_flags(FLAGS.__dict__)
  train()
