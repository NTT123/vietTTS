import haiku
import haiku as hk
import jax.numpy as jnp
import jax.random
from vietTTS.nat.config import FLAGS
from vietTTS.nat.model import AcousticModel


@hk.testing.transform_and_run
def test_duration():
    net = AcousticModel()
    token = jnp.zeros((2, 10), dtype=jnp.int32)
    lengths = jnp.zeros((2,), dtype=jnp.int32)
    durations = jnp.zeros((2, 10), dtype=jnp.float32)
    mel = jnp.zeros((2, 20, 160), dtype=jnp.float32)
    o1, o2 = net(token, mel, lengths, durations)
    assert o1.shape == (2, 20, 160)
    assert o2.shape == (2, 20, 160)
