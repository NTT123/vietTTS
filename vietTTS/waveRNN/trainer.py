import pickle
import time
from pathlib import Path
from typing import Deque

import IPython.display as ipd
from tabulate import tabulate
from vietTTS.tacotron.dsp import *

from .config import *
from .data_loader import *
from .model import *

net = hk.without_apply_rng(hk.transform_with_state(lambda x, m: WaveRNN()(x, m)))


def make_optim(lr):
  return optax.chain(
      optax.clip_by_global_norm(1.),
      optax.scale_by_adam(),
      optax.scale(-lr)
  )


def encode_16bit_mu_law(y, mu=255):
  y = y.astype(jnp.float32) / (2**15)  # [-1, 1]
  mu_y = jnp.sign(y) * jnp.log1p(mu * jnp.abs(y)) / jnp.log1p(mu)
  mu_y = ((mu_y + 1)/2 * 255).astype(jnp.int32)
  return mu_y


optimizer = make_optim(FLAGS._training_schedule[0].learning_rate)

data_iter = make_train_data_iter(FLAGS._training_schedule[0].batch_size)
next(data_iter).shape, next(data_iter).dtype

rng = jax.random.PRNGKey(42)
y = next(data_iter)
print('Data batch info:', y.shape, y.dtype)
mu = encode_16bit_mu_law(y)
sr = FLAGS.sample_rate

melfilter = MelFilter(sr, 1024, 80)
mel = melfilter(y.astype(jnp.float32)/(2**15))
pad = 1024
mu = mu[:, pad-1:-pad]
muinputs = mu[:, :-1]
params, aux = net.init(rng, muinputs, mel)
optim_state = optimizer.init(params)
training_step = 0


def loss_fn(params, aux, batch):
  melfilter = MelFilter(sr, 1024, 80)
  y = batch
  mu = encode_16bit_mu_law(y)
  y = y.astype(jnp.float32) / (2**15)
  mel = melfilter(y)
  pad = 1024
  mu = mu[:, (pad-1):-pad]
  muinputs = mu[:, :-1]
  mutargets = mu[:, 1:]
  logpr, aux = net.apply(params, aux, muinputs, mel)
  pr = jnp.exp(logpr)
  v = jnp.linspace(0, 255, 256)[None, None, :]
  mean = jnp.sum(pr * v, axis=-1, keepdims=True)
  variance = jnp.sum(jnp.square(v - mean) * pr, axis=-1, keepdims=True)
  reg = jnp.log(1 + jnp.sqrt(variance))
  targets = jax.nn.one_hot(mutargets, num_classes=256)
  llh = jnp.sum(targets * logpr, axis=-1)
  l1 = -jnp.mean(llh)
  l2 = 0.1 * jnp.mean(reg)
  return l1 + l2, (l1, l2, aux)


@jax.jit
def update(params, aux, optim_state, batch, learning_rate):
  (loss, (l1, l2, new_aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, aux, batch)
  optimizer = make_optim(learning_rate)
  updates, new_optim_state = optimizer.update(grads, optim_state)
  new_params = optax.apply_updates(params, updates)
  return (loss, l1, l2), new_params, new_aux, new_optim_state


ckpt_dir = FLAGS.ckpt_dir


def save_checkpoint(step, params, aux, optim_state, path=ckpt_dir):
  params, aux, optim_state = jax.device_get((params, aux, optim_state))
  fn = path / f'checkpoint_{step:08d}.pickle'
  print(f'  > saving checkpoint at {fn}')
  with open(fn, 'wb') as f:
    pickle.dump({'step': step, 'params': params, 'aux': aux, 'optim_state': optim_state}, f)


def load_latest_checkpoint(path=ckpt_dir):
  ckpts = sorted(path.glob('checkpoint_*.pickle'))
  if len(ckpts) == 0:
    print(f'There is no checkpoint in {path}')
    return None
  latest_ckpt = ckpts[-1]
  print(f"  Loading checkpoint from {latest_ckpt}")
  with open(latest_ckpt, 'rb') as f:
    dic = pickle.load(f)
  return dic


if not FLAGS.ckpt_dir.exists():
  FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)

def make_new_log_file():
  counter = len(tuple(FLAGS.ckpt_dir.glob('log.*.txt')))
  fn = FLAGS.ckpt_dir / f'log.{counter}.txt'
  print(f'Creating new log file at {fn}')
  return open(fn, 'w', buffering=1)


logfile = make_new_log_file()


dic = load_latest_checkpoint()

if dic is not None:
  training_step = dic['step']
  params = dic['params']
  aux = dic['aux']
  optim_state = dic['optim_state']


@hk.without_apply_rng
@hk.transform_with_state
def regenerate_from_signal_(y, rng):
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


regenerate_from_signal = jax.jit(regenerate_from_signal_.apply)


def gen_test_sample(test_clip):
  t1 = time.perf_counter()
  synthesized_clip, reg, pr = regenerate_from_signal(params, aux, test_clip, rng)[0]
  synthesized_clip = jax.device_get(synthesized_clip)
  synthesized_clip = librosa.mu_expand(synthesized_clip[0] - 128)
  t2 = time.perf_counter()
  delta = t2 - t1
  l = len(synthesized_clip) / sr
  print(f'  take {delta:.3f} seconds to generate {l:.3f} seconds, RTF = {delta / l:.3f}')
  plt.plot(jax.device_get(reg[0, :, 0]))
  plt.show()
  # plt.imshow(pr[0].T)
  # plt.show()
  plt.figure(figsize=(20, 6))
  plt.matshow(pr[0, 12000:12100].T, interpolation='nearest', fignum=0, aspect='auto')
  plt.show()
  # display(ipd.Audio(synthesized_clip, rate=sr))
  # display(ipd.Audio(test_clip[0], rate=sr))


test_clip, _sr = sf.read(test_wav_files[0], dtype='int16')
# gen_test_sample(test_clip=test_clip[None, :])


def print_flags(dict):
  values = [(k, v) for k, v in dict.items() if not k.startswith('_')]
  print(tabulate(values))


print_flags(FLAGS.__dict__)

losses = Deque(maxlen=100)
l1s = Deque(maxlen=100)
l2s = Deque(maxlen=100)
start = time.perf_counter()
training_config = FLAGS._training_schedule[0]


for step in range(training_step + 1, 1 + FLAGS.total_training_steps):
  training_step += 1
  # update train config
  if step % 1000 == 0:
    for config in FLAGS._training_schedule:
      if step < config.end_step:
        training_config = config
        data_iter = make_train_data_iter(training_config.batch_size)
        break

  (loss, l1, l2), params, aux, optim_state = update(params, aux,
                                                    optim_state, next(data_iter), training_config.learning_rate)
  l1s.append(l1)
  l2s.append(l2)

  if step % 100 == 0:
    end = time.perf_counter()
    speed = 100 / (end - start)
    start = end
    l1 = sum(l1s, 0.0).item() / len(l1s)
    l2 = sum(l2s, 0.0).item() / len(l2s)
    msg = f'  {step:06d} | train loss {l1:.3f} | reg loss {l2:.3f} | {speed:.3f} it/s | LR {training_config.learning_rate:.3e} | batch size {training_config.batch_size} '
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
    gen_test_sample(test_clip=test_clip[None, :])
    # gen_test_sample(test_clip=test_clip[None, ss:ss+l])
    # gen_test_sample()
