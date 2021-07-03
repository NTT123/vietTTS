import pickle
from functools import partial
from typing import Deque

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm
from vietTTS.nat.config import AcousticInput

from .config import FLAGS, AcousticInput
from .data_loader import load_textgrid_wav
from .dsp import MelFilter
from .model import NATNet
from .utils import print_flags


@hk.transform_with_state
def net(x): return NATNet(is_training=True)(x)


@hk.transform_with_state
def val_net(x): return NATNet(is_training=False)(x)


def loss_fn(params, aux, rng, inputs: AcousticInput, is_training=True):
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim, FLAGS.fmin, FLAGS.fmax)
  mels = melfilter(inputs.wavs.astype(jnp.float32) / (2**15))
  B, L, D = mels.shape
  inp_mels = jnp.concatenate((jnp.zeros((B, 1, D), dtype=jnp.float32), mels[:, :-1, :]), axis=1)
  inputs = inputs._replace(mels=inp_mels)
  (mel1_hat, mel2_hat, duration_hat), new_aux = (net if is_training else val_net).apply(params, aux, rng, inputs)
  loss = (jnp.abs(mel1_hat - mels) + jnp.abs(mel2_hat - mels)) / 2
  loss = jnp.mean(loss, axis=-1)
  mask = jnp.arange(0, L)[None, :] < (inputs.wav_lengths // FLAGS.hop_length)[:, None]
  loss = jnp.sum(loss * mask) / jnp.sum(mask)
  duration_loss = jnp.abs(duration_hat - inputs.durations)
  B, L = duration_loss.shape
  mask = jnp.arange(0, L)[None, :] < inputs.lengths[:, None]
  # NOT predict [WORD END] token
  mask = jnp.where(inputs.phonemes == FLAGS.word_end_index, False, mask)
  duration_loss = jnp.sum(duration_loss * mask) / jnp.sum(mask)
  loss = loss + duration_loss
  return (loss, new_aux) if is_training else (loss, new_aux, mel2_hat, mels, duration_hat, inputs.durations)


train_loss_fn = partial(loss_fn, is_training=True)
val_loss_fn = jax.jit(partial(loss_fn, is_training=False))

loss_vag = jax.value_and_grad(train_loss_fn, has_aux=True)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
)


@jax.jit
def update(params, aux, rng, optim_state, inputs):
  rng, new_rng = jax.random.split(rng)
  (loss, new_aux), grads = loss_vag(params, aux, rng, inputs)
  updates, new_optim_state = optimizer.update(grads, optim_state, params)
  new_params = optax.apply_updates(updates, params)
  return loss, (new_params, new_aux, new_rng, new_optim_state)


def initial_state(batch):
  rng = jax.random.PRNGKey(42)
  params, aux = hk.transform_with_state(lambda x: NATNet(True)(x)).init(rng, batch)
  optim_state = optimizer.init(params)
  return params, aux, rng, optim_state


def plot_val_duration(step: int, predicted_dur, gt_dur, length: int):
  fn = FLAGS.ckpt_dir / f'duration_{step}.png'
  plt.plot(predicted_dur[:length])
  plt.plot(gt_dur[:length])
  plt.legend(['predicted', 'gt'])
  plt.title("Phoneme durations")
  plt.savefig(fn)
  plt.close()


def train():
  train_data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len,
                                      FLAGS.batch_size, FLAGS.max_wave_len, 'train')
  val_data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len,
                                    FLAGS.batch_size, FLAGS.max_wave_len, 'val')
  melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim, FLAGS.fmin, FLAGS.fmax)
  batch = next(train_data_iter)
  batch = batch._replace(mels=melfilter(batch.wavs.astype(jnp.float32) / (2**15)))
  params, aux, rng, optim_state = initial_state(batch)
  losses = Deque(maxlen=1000)
  val_losses = Deque(maxlen=100)

  last_step = -1

  # loading latest checkpoint
  ckpt_fn = FLAGS.ckpt_dir / 'nat_ckpt_latest.pickle'
  if ckpt_fn.exists():
    print('Resuming from latest checkpoint at', ckpt_fn)
    with open(ckpt_fn, 'rb') as f:
      dic = pickle.load(f)
      last_step, params, aux, rng, optim_state = dic['step'], dic['params'], dic['aux'], dic['rng'], dic['optim_state']

  tr = tqdm(
      range(last_step + 1, FLAGS.num_training_steps + 1),
      desc='training',
      total=FLAGS.num_training_steps+1,
      initial=last_step+1
  )
  for step in tr:
    batch = next(train_data_iter)
    loss, (params, aux, rng, optim_state) = update(params, aux, rng, optim_state, batch)
    losses.append(loss)

    if step % 10 == 0:
      val_batch = next(val_data_iter)
      val_loss, val_aux, predicted_mel, gt_mel, duration_hat, duration_gt = val_loss_fn(params, aux, rng, val_batch)
      val_losses.append(val_loss)
      attn = jax.device_get(val_aux['nat_net']['attn'][0])
      predicted_mel = jax.device_get(predicted_mel[0])
      gt_mel = jax.device_get(gt_mel[0])
      duration_hat = jax.device_get(duration_hat[0])
      duration_gt = jax.device_get(duration_gt[0])
      token_length = val_batch.lengths[0]

    if step % 1000 == 0:
      loss = sum(losses).item() / len(losses)
      val_loss = sum(val_losses).item() / len(val_losses)
      tr.write(f'step {step}  train loss {loss:.3f}  val loss {val_loss:.3f}')

      plot_val_duration(step, duration_hat, duration_gt, token_length)

      # saving predicted mels
      plt.figure(figsize=(10, 10))
      plt.subplot(3, 1, 1)
      plt.imshow(predicted_mel.T, origin='lower', aspect='auto')
      plt.subplot(3, 1, 2)
      plt.imshow(gt_mel.T, origin='lower', aspect='auto')
      plt.subplot(3, 1, 3)
      plt.imshow(attn.T, origin='lower', aspect='auto')
      plt.tight_layout()
      plt.savefig(FLAGS.ckpt_dir / f'mel_{step}.png')
      plt.close()

      # saving checkpoint
      with open(ckpt_fn, 'wb') as f:
        pickle.dump({'step': step, 'params': params, 'aux': aux, 'rng': rng, 'optim_state': optim_state}, f)


if __name__ == '__main__':
  print_flags(FLAGS.__dict__)
  if not FLAGS.ckpt_dir.exists():
    print('Create checkpoint dir at', FLAGS.ckpt_dir)
    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
  train()
