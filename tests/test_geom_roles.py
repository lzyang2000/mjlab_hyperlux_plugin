"""CPU-only checks for HyperLux geom role selection."""

import mujoco
import numpy as np
import pytest
import warp as wp

from mjlab_hyperlux import HyperLuxCameraSensorCfg
from mjlab_hyperlux.renderer import _resolve_batch_size, _resolve_fidelity, _split_geoms


def _model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body name="box_body">
          <geom name="box" type="box" size=".1 .1 .1"/>
        </body>
      </worldbody>
    </mujoco>
    """
  )


def test_geom_roles_support_globs_and_preserve_unmatched_dynamic_behavior():
  model = _model()
  static, dynamic = _split_geoms(model, np.arange(model.ngeom), ("floor",), ("box*",))

  np.testing.assert_array_equal(static, [0])
  np.testing.assert_array_equal(dynamic, [1])


def test_geom_roles_reject_overlapping_labels():
  model = _model()

  with pytest.raises(ValueError, match="matches both"):
    _split_geoms(model, np.arange(model.ngeom), ("*",), ("box",))


def test_fidelity_keeps_legacy_shadow_defaults_and_allows_override():
  plain = HyperLuxCameraSensorCfg(name="camera", camera_name="cam")
  shadows = HyperLuxCameraSensorCfg(name="camera", camera_name="cam", use_shadows=True)
  low = HyperLuxCameraSensorCfg(
    name="camera", camera_name="cam", hyperlux_fidelity="low"
  )

  assert _resolve_fidelity(plain) == "high"
  assert _resolve_fidelity(shadows) == "veryhigh"
  assert _resolve_fidelity(low) == "low"


def test_auto_batch_has_a_safe_cpu_fallback_and_validates_overrides():
  device = wp.get_device("cpu")
  cfg = HyperLuxCameraSensorCfg(name="camera", camera_name="cam")
  assert _resolve_batch_size(cfg, device) == 128
  assert (
    _resolve_batch_size(
      HyperLuxCameraSensorCfg(name="camera", camera_name="cam", hyperlux_batch_size=7),
      device,
    )
    == 7
  )
  with pytest.raises(ValueError, match="positive"):
    _resolve_batch_size(
      HyperLuxCameraSensorCfg(name="camera", camera_name="cam", hyperlux_batch_size=0),
      device,
    )
