import haiku
import haiku as hk
import jax.numpy as jnp
import jax.random
from vietTTS.nat.config import FLAGS
from vietTTS.nat.model import DurationModel


@hk.testing.transform_and_run
def test_duration():
    net = DurationModel()
    p = jnp.zeros((2, 10), dtype=jnp.int32)
    l = jnp.zeros((2,), dtype=jnp.int32)
    o = net(p, l)
    assert o.shape == (2, 10, 1)
