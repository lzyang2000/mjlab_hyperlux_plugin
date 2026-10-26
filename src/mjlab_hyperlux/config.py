"""HyperLux-specific camera options; shared camera behavior stays in mjlab."""

from dataclasses import dataclass
from typing import Literal

from mjlab.sensor import CameraSensorCfg


@dataclass
class HyperLuxCameraSensorCfg(CameraSensorCfg):
  rgb_backend: str = "hyperlux"
  """Registered RGB backend. Depth and segmentation still use MuJoCo Warp."""

  hyperlux_anti_aliasing_grid: Literal[1, 2, 3, 4] = 1
  """Samples per pixel along each axis; 2 gives 2 x 2 supersampling."""

  hyperlux_batch_size: int | Literal["auto"] = "auto"
  """Worlds rendered per submission, or ``"auto"`` for batch 128."""

  hyperlux_background: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Linear RGB background visible through openings and in reflected rays."""

  hyperlux_fidelity: Literal["low", "medium", "high", "veryhigh"] | None = None
  """HyperLux shading and geometry fidelity.

  ``None`` preserves the legacy selection: ``"veryhigh"`` with shadows and
  ``"high"`` otherwise.
  """

  hyperlux_static_geoms: tuple[str, ...] = ()
  """Geom names or glob patterns rendered in the shared HyperLux scene.

  A static geom must have the same world-space pose in every environment and
  must not move after initialization. Unmatched geoms keep the legacy dynamic
  overlay behavior. Names are the compiled MuJoCo names (for example,
  ``"world/floor"`` or ``"world/*"``).
  """

  hyperlux_dynamic_geoms: tuple[str, ...] = ()
  """Geom names or glob patterns forced into per-environment dynamic overlays.

  Use this for environment-specific terrain, obstacles, or randomized objects,
  even when each individual object is stationary during a rollout.
  """
