"""
A script to train acoustic model.

Usage:
  python3 -m vietTTS.nat.acoustic_trainer

"""
import os
import pickle
from functools import partial
from typing import Deque, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm
from vietTTS.nat.config import AcousticInput

from .config import FLAGS, AcousticInput
from .data_loader import load_textgrid_wav
from .dsp import MelFilter
from .model import AcousticModel
from .utils import print_flags

if 'COLAB_TPU_ADDR' in os.environ:
  print("Setting up TPU cores")
  jax.tools.colab_tpu.setup_tpu()
print("Jax devices:", jax.devices())


@hk.transform_with_state
def net(x): return AcousticModel(is_training=True)(x)


@hk.transform_with_state
def val_net(x): return AcousticModel(is_training=False)(x)


melfilter = MelFilter(FLAGS.sample_rate, FLAGS.n_fft, FLAGS.mel_dim, FLAGS.fmin, FLAGS.fmax)
hop_length = FLAGS.n_fft // 4
num_devices = jax.device_count()


def loss_fn(params, aux, rng, inputs: AcousticInput, is_training=True):
  """
    Compute loss: (l1_loss + l2_loss) / 2
  """
  mels = melfilter(inputs.wavs.astype(jnp.float32) / (2**15))
  mels = (mels - FLAGS.data_mean) / FLAGS.data_std
  B, L, D = mels.shape
  inp_mels = jnp.concatenate((jnp.zeros((B, 1, D), dtype=jnp.float32), mels[:, :-1, :]), axis=1)
  n_frames = inputs.durations * FLAGS.sample_rate / hop_length
  inputs = inputs._replace(mels=inp_mels, durations=n_frames)
  (mel1_hat, mel2_hat), new_aux = (net if is_training else val_net).apply(params, aux, rng, inputs)

  # l1 + l2
  loss1 = (jnp.square(mel1_hat - mels) + jnp.square(mel2_hat - mels)) / 2
  loss2 = (jnp.abs(mel1_hat - mels) + jnp.abs(mel2_hat - mels)) / 2
  loss = jnp.mean((loss1 + loss2)/2, axis=-1)
  # compute masks for mel targets
  mask = jnp.arange(0, L)[None, :] < (inputs.wav_lengths // hop_length)[:, None]
  # compute loss per element
  loss = jnp.sum(loss * mask) / jnp.sum(mask)
  return (loss, new_aux) if is_training else (loss, new_aux, mel2_hat, mels)


train_loss_fn = partial(loss_fn, is_training=True)
val_loss_fn = jax.pmap(partial(loss_fn, is_training=False), axis_name='i')
loss_vag = jax.value_and_grad(train_loss_fn, has_aux=True)

lr_scheduler = optax.warmup_exponential_decay_schedule(
    0.0, FLAGS.learning_rate, 1_000, 50_000, 0.5, 0, False, FLAGS.learning_rate/100)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(lr_scheduler, weight_decay=FLAGS.weight_decay)
)


def update_step(prev_state, inputs):
  params, aux, rng, optim_state = prev_state
  rng, new_rng = jax.random.split(rng)
  (loss, new_aux), grads = loss_vag(params, aux, rng, inputs)
  updates, new_optim_state = optimizer.update(grads, optim_state, params)
  new_params = optax.apply_updates(updates, params)
  return (new_params, new_aux, new_rng, new_optim_state), loss


@partial(jax.pmap, axis_name='i')
def update(params, aux, rng, optim_state, inputs):
  new_state, losses = jax.lax.scan(update_step, (params, aux, rng, optim_state), inputs)
  return jnp.mean(losses), new_state


def add_new_dims(x: jnp.ndarray, dims: Sequence[int]) -> jnp.ndarray:
  """reshape an array by adding new dimensions"""
  return jax.tree_map(lambda x: jnp.reshape(x, dims + (-1,) + x.shape[1:]), x)


def initial_state(batch):
  rng = jax.random.PRNGKey(42)
  params, aux = hk.transform_with_state(lambda x: AcousticModel(True)(x)).init(rng, batch)
  optim_state = optimizer.init(params)
  return params, aux, rng, optim_state


def train():
  train_data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len,
                                      FLAGS.batch_size, FLAGS.max_wave_len, 'train')
  val_data_iter = load_textgrid_wav(FLAGS.data_dir, FLAGS.max_phoneme_seq_len,
                                    FLAGS.batch_size, FLAGS.max_wave_len, 'val')
  batch = next(train_data_iter)
  batch = batch._replace(mels=melfilter(batch.wavs.astype(jnp.float32) / (2**15)))

  mel100 = batch.mels[:, :100]
  data_mean = jnp.mean(mel100)
  data_std = jnp.std(mel100)
  print(
      f'''Statistics of a batch vs FLAGS: mean/FLAGS.mean {data_mean}/{FLAGS.data_mean}  std/FLAGS.std {data_std}/{FLAGS.data_std}. 
      Modify config.py if these values do not matched!''')
  losses = Deque(maxlen=1000)
  val_losses = Deque(maxlen=100)

  spu = FLAGS.steps_per_update
  last_step = -spu

  # loading latest checkpoint
  ckpt_fn = FLAGS.ckpt_dir / 'acoustic_ckpt_latest.pickle'
  if ckpt_fn.exists():
    print('Resuming from latest checkpoint at', ckpt_fn)
    with open(ckpt_fn, 'rb') as f:
      dic = pickle.load(f)
      last_step, params, aux, rng, optim_state = dic['step'], dic['params'], dic['aux'], dic['rng'], dic['optim_state']
  else:
    params, aux, rng, optim_state = initial_state(batch)
  params, aux, rng, optim_state = jax.device_put_replicated((params, aux, rng, optim_state), jax.devices())

  tr = tqdm(range(last_step + spu, FLAGS.num_training_steps + spu, spu),
            desc='training',
            total=FLAGS.num_training_steps//spu+1,
            initial=last_step//spu+1,
            unit_scale=spu)
  for step in tr:
    batch = add_new_dims(next(train_data_iter), (num_devices, spu))

    loss, (params, aux, rng, optim_state) = update(params, aux, rng, optim_state, batch)
    losses.append(loss)

    if step % 10 == 0:
      val_batch = next(val_data_iter)
      val_loss, val_aux, predicted_mel, gt_mel = val_loss_fn(params, aux, rng, val_batch)
      val_losses.append(val_loss)

    if step % 1000 == 0:
      loss = jnp.mean(sum(losses)).item() / len(losses)
      val_loss = jnp.mean(sum(val_losses)).item() / len(val_losses)
      tr.write(f'step {step}  train loss {loss:.3f}  val loss {val_loss:.3f}')

      # saving predicted mels
      attn = jax.device_get(val_aux['acoustic_model']['attn'][0, 0])
      predicted_mel = jax.device_get(predicted_mel[0, 0])
      gt_mel = jax.device_get(gt_mel[0, 0])
      plt.figure(figsize=(10, 10))
      plt.subplot(3, 1, 1)
      min_value = jnp.min(gt_mel.T).item()
      max_value = jnp.max(gt_mel.T).item()
      plt.imshow(predicted_mel.T, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
      plt.subplot(3, 1, 2)
      plt.imshow(gt_mel.T, origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
      plt.subplot(3, 1, 3)
      plt.imshow(attn.T, origin='lower', aspect='auto')
      plt.tight_layout()
      plt.savefig(FLAGS.ckpt_dir / f'mel_{step}.png')
      plt.close()

      # saving checkpoint
      with open(ckpt_fn, 'wb') as f:
        params_, aux_, rng_, optim_state_ = jax.tree_map(lambda x: jnp.device_get(x[0]),
                                                         (params, aux, rng, optim_state))
        pickle.dump({'step': step, 'params': params_, 'aux': aux_, 'rng': rng_, 'optim_state': optim_state_}, f)


if __name__ == '__main__':
  print_flags(FLAGS.__dict__)
  if not FLAGS.ckpt_dir.exists():
    print('Create checkpoint dir at', FLAGS.ckpt_dir)
    FLAGS.ckpt_dir.mkdir(parents=True, exist_ok=True)
  train()
