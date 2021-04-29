import haiku as hk
import jax
import jax.numpy as jnp

from .config import *


class UpsampleNetwork(hk.Module):
  def __init__(self, num_output_channels, is_training=True):
    super().__init__()
    self.input_conv = hk.Conv1D(512, 3, padding='VALID')
    self.input_bn = hk.BatchNorm(True, True, 0.99)
    self.dilated_conv_1 = hk.Conv1D(512, 2, 1, rate=2, padding='VALID')
    self.dilated_bn_1 = hk.BatchNorm(True, True, 0.99)
    self.dilated_conv_2 = hk.Conv1D(512, 2, 1, rate=4, padding='VALID')
    self.dilated_bn_2 = hk.BatchNorm(True, True, 0.99)

    self.upsample_conv_1 = hk.Conv1DTranspose(512, kernel_shape=1, stride=2, padding='SAME')
    self.upsample_bn_1 = hk.BatchNorm(True, True, 0.99)
    self.upsample_conv_2 = hk.Conv1DTranspose(512, kernel_shape=1, stride=2, padding='SAME')
    self.upsample_bn_2 = hk.BatchNorm(True, True, 0.99)
    self.upsample_conv_3 = hk.Conv1DTranspose(num_output_channels, kernel_shape=1, stride=4, padding='SAME')
    self.upsample_bn_3 = hk.BatchNorm(True, True, 0.99)
    self.is_training = is_training

  def __call__(self, mel):
    x = jax.nn.relu(self.input_bn(self.input_conv(mel), is_training=self.is_training))
    res_1 = jax.nn.relu(self.dilated_bn_1(self.dilated_conv_1(x), is_training=self.is_training))
    x = x[:, 1:-1] + res_1
    res_2 = jax.nn.relu(self.dilated_bn_2(self.dilated_conv_2(x), is_training=self.is_training))
    x = x[:, 2:-2] + res_2

    x = jax.nn.relu(self.upsample_bn_1(self.upsample_conv_1(x), is_training=self.is_training))
    x = jax.nn.relu(self.upsample_bn_2(self.upsample_conv_2(x), is_training=self.is_training))
    x = jax.nn.relu(self.upsample_bn_3(self.upsample_conv_3(x), is_training=self.is_training))

    # tile x16
    N, L, D = x.shape
    x = jnp.tile(x[:, :, None, :], (1, 1, 16, 1))
    x = jnp.reshape(x, (N, -1, D))

    return x


class WaveRNN(hk.Module):
  def __init__(self, is_training=True):
    super().__init__()
    self.gru = hk.GRU(FLAGS.gru_dim)
    self.input_embed = hk.Embed(256, FLAGS.embed_dim)
    self.o1 = hk.Linear(FLAGS.gru_dim)
    self.o2 = hk.Linear(256)
    self.upsample = UpsampleNetwork(num_output_channels=FLAGS.embed_dim, is_training=is_training)
    self.is_training = is_training

  def __call__(self, x, mel):
    x = self.input_embed(x) + self.upsample(mel)
    x, hx = hk.dynamic_unroll(
        self.gru,
        x,
        self.gru.initial_state(x.shape[0]),
        time_major=False
    )
    logits = self.o2(jax.nn.relu(self.o1(x)))
    logpr = jax.nn.log_softmax(logits)
    return logpr
