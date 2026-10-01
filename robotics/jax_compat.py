"""Compatibility shim: Brax 0.14.2 still calls jax.device_put_replicated /
jax.device_put_sharded, which JAX >= 0.10 removed (Flax 0.12.10 needs JAX >= 0.11.1).
On a single host the drop-in replacement is stacking and placing the copies."""
import jax
import jax.numpy as jnp


def _put_replicated(x, devices):
    n = len(devices)
    return jax.tree.map(lambda a: jax.device_put(jnp.stack([jnp.asarray(a)] * n), devices[0]) if n == 1
                        else jax.device_put_sharded([a] * n, devices), x)


def _put_sharded(shards, devices):
    return jax.tree.map(lambda *xs: jax.device_put(jnp.stack(xs), devices[0]), *shards)


for name, fn in (("device_put_replicated", _put_replicated), ("device_put_sharded", _put_sharded)):
    try:
        getattr(jax, name)
    except AttributeError:
        setattr(jax, name, fn)
