"""Silence MuJoCo Warp's per-world solver / line-search "iterations limit reached" printfs.

MJX's Warp backend sets opt.warn_overflow = ALL, so every simulated world that
hits the line-search cap prints two lines from inside a GPU kernel. With 8192
worlds that is gigabytes of log per training job and a large slowdown. The cap
itself (opt.ls_iterations from the model XML) is unchanged; MJX's JAX backend
applies the same cap silently. Only the ITERATIONS and LS_ITERATIONS bits are cleared (the model sets iterations=1, ls_iterations=5 on purpose); all other
overflow warnings (contacts, constraints, broadphase...) stay on.
"""
from mujoco.mjx._src import io as _mjx_io
from mujoco.mjx.third_party.mujoco_warp._src import types as _wt

_orig = _mjx_io.mjwp.put_model


def _put_model(*args, **kwargs):
    mw = _orig(*args, **kwargs)
    mw.opt.warn_overflow = int(mw.opt.warn_overflow) & ~int(_wt.OverflowType.LS_ITERATIONS) & ~int(_wt.OverflowType.ITERATIONS)
    return mw


_mjx_io.mjwp.put_model = _put_model
