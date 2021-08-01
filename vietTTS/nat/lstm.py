from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from haiku import LSTMState, RNNCore


class LSTM(RNNCore):
  def __init__(self, hidden_size: int, name: Optional[str] = None):
    super().__init__(name=name)
    self.hidden_size = hidden_size

  def __call__(
      self,
      inputs: jnp.ndarray,
      prev_state: LSTMState,
      h_mask=None,
      c_mask=None
  ) -> Tuple[jnp.ndarray, LSTMState]:
    if len(inputs.shape) > 2 or not inputs.shape:
      raise ValueError("LSTM input must be rank-1 or rank-2.")
    x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
    w_init = hk.initializers.RandomUniform(-0.1, 0.1)
    gated = hk.Linear(4 * self.hidden_size, w_init=w_init)(x_and_h)
    # TODO(slebedev): Consider aligning the order of gates with Sonnet.
    # i = input, g = cell_gate, f = forget_gate, o = output_gate
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
    f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
    c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
    c = jnp.clip(c, a_min=-10, a_max=10)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    if h_mask is not None:
      h = jnp.where(h_mask, prev_state.hidden, h)
    if c_mask is not None:
      c = jnp.where(c_mask, prev_state.cell, c)
    return h, LSTMState(h, c)

  def initial_state(self, batch_size: int) -> LSTMState:
    state = LSTMState(hidden=jnp.zeros([batch_size, self.hidden_size]),
                      cell=jnp.zeros([batch_size, self.hidden_size]))
    return state
