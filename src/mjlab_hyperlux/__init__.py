"""Opt-in HyperLux RGB camera configuration for mjlab.

The native renderer is JIT-compiled.  Keep its generated build tree in the
system temporary directory by default so importing this plugin does not leave
machine-specific build artifacts in a user's home directory.  Set
``HYPERLUX_CACHE_DIR`` before importing the package to use a persistent cache.
"""

import os

from mjlab_hyperlux.config import HyperLuxCameraSensorCfg as HyperLuxCameraSensorCfg
from mjlab_hyperlux.lighting import (
  LIGHTING_PROFILES,
  LightingProfile,
  RendererLightingScale,
  apply_mujoco_lighting,
  get_lighting_profile,
)

os.environ.setdefault("HYPERLUX_CACHE_DIR", "/tmp/hyperlux")

__all__ = [
  "HyperLuxCameraSensorCfg",
  "LIGHTING_PROFILES",
  "LightingProfile",
  "RendererLightingScale",
  "apply_mujoco_lighting",
  "get_lighting_profile",
]
