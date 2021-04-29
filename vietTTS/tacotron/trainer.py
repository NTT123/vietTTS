import os
import pickle
import time
from functools import partial
from typing import Deque

import matplotlib.pyplot as plt

from .config import *
from .data_loader import *
from .dsp import *
from .model import *


def train_tacotron_forward(x, r, m): return Tacotron(is_training=True)(x, r, m)
def test_tacotron_forward(x, r, m): return Tacotron(is_training=False)(x, r, m)


net = hk.transform_with_state(train_tacotron_forward)


def loss_fn(params, aux, rng, inputs: InputBatch, reduce_factor, mel_dropout):
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim)
  mel = melfilter(inputs.wav.astype(jnp.float32) / (2**15))
  N, L, D = mel.shape
  # import pdb; pdb.set_trace()
  input_mel = mel[:, (reduce_factor-1):(-reduce_factor):reduce_factor, :]
  input_mel = jnp.concatenate((
      jnp.zeros((N, 1, D)),
      input_mel
  ), axis=1)
  target_mel = mel[:, :(input_mel.shape[1] * reduce_factor), :]
  # input_mel, target_mel = mel[:, :-1], mel[:, 1:]
  output, new_aux = net.apply(params, aux, rng, TacotronInput(
      inputs.text, inputs.text_len, input_mel), reduce_factor, mel_dropout)

  def l2_loss(a, b): return jnp.square(a-b)
  def l1_loss(a, b): return jnp.abs(a-b)
  def loss_(a, b): return (l1_loss(a, b) + l2_loss(a, b)) / 2.
  loss = (loss_(output.mel, target_mel) + loss_(output.mel + output.mel_residual, target_mel)) / 2.
  loss = jnp.mean(loss, axis=-1)
  N, L = loss.shape
  eoc = (jnp.arange(0, L)[None, :] >= inputs.mel_len[:, None]).astype(jnp.int32)
  loss2 = -(
      jax.nn.log_sigmoid(output.stop_token[..., 0]) * eoc +
      jax.nn.log_sigmoid(-output.stop_token[..., 0]) * (1-eoc)
  )
  loss = jnp.mean(loss + loss2)
  # loss = jnp.mean(loss)
  # loss = mask * loss
  # loss = jnp.sum(loss) / jnp.sum(mask)

  return loss, new_aux


# linear warmup
def schedule_fn(x): return jnp.clip(x/10_000, a_min=0., a_max=1.)


def make_optim(lr):
  return optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.scale_by_rms(),
      optax.scale(-lr),
      optax.scale_by_schedule(schedule_fn),
  )


optimizer = make_optim(FLAGS._training_schedule[0].learning_rate)


@partial(jax.jit, static_argnums=[5, 6, 7])
def update(params, aux, rng, optim_state, inputs, reduce_factor, learning_rate, mel_dropout):
  rng, rng_next = jax.random.split(rng)
  vag = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, new_aux), grads = vag(params, aux, rng, inputs, reduce_factor, mel_dropout)
  optimizer = make_optim(learning_rate)
  updates, new_optim_state = optimizer.update(grads, optim_state, params)
  new_params = optax.apply_updates(params, updates)
  return loss, new_params, new_aux, new_optim_state, rng_next


train_iter = load_text_wav(FLAGS.data_dir, FLAGS.batch_size, 16000*10, 256)
gta_iter = load_text_wav_name(FLAGS.data_dir, FLAGS.batch_size, 16000*10, 256)

batch = next(train_iter)

rng = jax.random.PRNGKey(42)
rng1, rng2, rng3 = jax.random.split(rng, 3)
text = jax.random.randint(rng1, (3, 10), 0, 256)
text_len = jax.random.randint(rng2, (3,), 3, 9)
mel = jax.random.normal(rng3, (3, 20, 80))
inputs = TacotronInput(text, text_len, mel)
params, aux = net.init(rng, inputs, 1, 0.5)
optim_state = optimizer.init(params)


def plot_attn(L, LL, step=0):
  plt.figure(figsize=(10, 3))
  plt.imshow(
      jax.device_get(aux['tacotron']['attn'])[:L, :LL],
      interpolation='nearest',
      aspect='auto',
  )
  plt.savefig(FLAGS.ckpt_dir / f'attn_{step}.png')
  plt.close()


if not FLAGS.ckpt_dir.exists():
  FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)


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


last_step = -1
ckpt = load_latest_ckpt()
if ckpt is None:
  print(f"There is no checkpoint at {FLAGS.ckpt_dir}")
else:
  print(f"Loading checkpoint at {FLAGS.ckpt_dir}")
  last_step, params, aux, optim_state, rng = ckpt


def make_new_log_file():
  counter = len(tuple(FLAGS.ckpt_dir.glob('log.*.txt')))
  fn = FLAGS.ckpt_dir / f'log.{counter}.txt'
  print(f'Creating new log file at {fn}')
  return open(fn, 'w', buffering=1)


logfile = make_new_log_file()


losses = Deque(maxlen=1000)

start = time.perf_counter()


def print_flags(dict):
  values = [(k, v) for k, v in dict.items() if not k.startswith('_')]
  print(tabulate(values))


print_flags(FLAGS.__dict__)


current_config = FLAGS._training_schedule[0]
for config in FLAGS._training_schedule:
  if last_step + 1 < config.end_step:
    R = config.reduce_factor
    current_config = config
    break


for i in range(last_step + 1, 1 + FLAGS.training_steps):
  last_step += 1

  # update learning config
  if i % 100 == 0:
    for config in FLAGS._training_schedule:
      if i < config.end_step:
        current_config = config
        break

  batch = next(train_iter)

  pad_mel = (i % (current_config.reduce_factor//2 + 1))
  N, L = batch.wav.shape
  pad = pad_mel * (FLAGS.n_fft//4)
  wav = np.pad(
      batch.wav[:, pad:],
      ((0, 0), (0, pad)),
      mode='edge'
  )
  batch = InputBatch(
      batch.text,
      batch.text_len,
      wav,
      batch.mel_len - pad_mel
  )

  loss, params, aux, optim_state, rng = update(
      params,
      aux,
      rng,
      optim_state,
      batch,
      current_config.reduce_factor,
      current_config.learning_rate,
      current_config.mel_dropout
  )
  losses.append(loss)

  if i % 1000 == 0:
    save_ckpt(i, params, aux, optim_state, rng)

  if i % FLAGS.logging_freq == 0:
    end = time.perf_counter()
    speed = FLAGS.logging_freq / (end - start)
    start = end
    loss = sum(losses, 0.0).item() / len(losses)
    lr = schedule_fn(optim_state[-1].count).item() * current_config.learning_rate
    msg = f'  {i:06d} | loss {loss:.3f} | {speed:.3f} it/s | LR {lr:.3e} | reduce factor {current_config.reduce_factor} | mel dropout {current_config.mel_dropout:.3f} '
    logfile.write(msg + '\n')
    print(msg)

  if i % (10*FLAGS.logging_freq) == 0:
    plot_attn(batch.text_len[0], batch.mel_len[0]//current_config.reduce_factor, step=i)
