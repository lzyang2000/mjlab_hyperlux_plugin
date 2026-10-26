"""CPU checks for renderer-neutral lighting profiles."""

import mujoco
import numpy as np
import pytest

from mjlab_hyperlux import apply_mujoco_lighting, get_lighting_profile
from mjlab_hyperlux.lighting import estimate_point_bounce


def _model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <worldbody>
        <light name="sun" type="directional" diffuse="1 1 1"/>
        <light name="lamp" type="point" diffuse="1 1 1"/>
      </worldbody>
    </mujoco>
    """
  )


def test_indoor_profile_scales_hyperlux_local_light_and_ambient():
  model = _model()
  original_diffuse = model.light_diffuse.copy()
  original_ambient = model.light_ambient.copy()

  apply_mujoco_lighting(model, get_lighting_profile("indoor"), "hyperlux")

  np.testing.assert_allclose(model.light_diffuse[0], original_diffuse[0] * 1.0)
  np.testing.assert_allclose(model.light_diffuse[1], original_diffuse[1] * 3.8)
  np.testing.assert_allclose(model.light_ambient, original_ambient * 0.7)
  np.testing.assert_allclose(model.light_attenuation[1], [0, 0, 1])


def test_unknown_profile_is_actionable():
  with pytest.raises(ValueError, match="choose from"):
    get_lighting_profile("warehouse")


def _room(
  scale: float = 1, ceiling: bool = True, reflectance: float = 0.6
) -> mujoco.MjModel:
  walls = [
    ((0, 0, -2), (2, 2, 0.05)),
    ((0, 0, 2), (2, 2, 0.05)),
    ((-2, 0, 0), (0.05, 2, 2)),
    ((2, 0, 0), (0.05, 2, 2)),
    ((0, -2, 0), (2, 0.05, 2)),
    ((0, 2, 0), (2, 0.05, 2)),
  ]
  geoms = "".join(
    f'<geom type="box" pos="{" ".join(str(v * scale) for v in pos)}" '
    f'size="{" ".join(str(v * scale) for v in size)}" material="wall"/>'
    for index, (pos, size) in enumerate(walls)
    if ceiling or index != 1
  )
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <asset><material name="wall" rgba="{reflectance} {reflectance} {reflectance} 1" metallic="0"/></asset>
    <worldbody><light type="point" diffuse="1 1 1" attenuation="0 0 1"/>{geoms}</worldbody>
  </mujoco>''')


def test_bounce_obeys_room_scale_and_rough_enclosure_solution():
  small = estimate_point_bounce(_room())
  large = estimate_point_bounce(_room(scale=2))
  # Inner cube faces are 3.9 m apart; the diffuse enclosure solution is 4 I rho / (A (1-rho)).
  expected = 4 * 0.6 / (6 * 3.9**2 * (1 - 0.6))
  np.testing.assert_allclose(small, expected, rtol=0.04)
  np.testing.assert_allclose(large, small / 4, rtol=0.01)


def test_bounce_responds_to_openings_reflectance_and_light_state():
  closed = estimate_point_bounce(_room())
  assert (estimate_point_bounce(_room(ceiling=False)) < closed).all()
  assert (estimate_point_bounce(_room(reflectance=0.1)) < closed).all()
  dark = _room()
  dark.light_active[:] = False
  np.testing.assert_array_equal(estimate_point_bounce(dark), np.zeros(3))
  np.testing.assert_array_equal(
    estimate_point_bounce(_room(reflectance=0)), np.zeros(3)
  )


def test_bounce_does_not_modify_input_geometry_or_reduce_authored_ambient():
  model = _room()
  groups = model.geom_group.copy()
  model.light_ambient[:] = 0.5
  apply_mujoco_lighting(model, get_lighting_profile("indoor"), "hyperlux")
  np.testing.assert_array_equal(model.geom_group, groups)
  np.testing.assert_allclose(model.light_ambient, 0.35)


def test_mocap_room_is_sampled_and_articulated_prop_is_ignored():
  # Reproduce mjlab's fixed scene attached under a mocap entity root.
  spec = mujoco.MjSpec()
  room = _room()
  root = spec.worldbody.add_body(name="room", mocap=True)
  for i in range(room.ngeom):
    root.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=room.geom_pos[i],
      size=room.geom_size[i],
      rgba=[0.6, 0.6, 0.6, 1],
    )
  spec.worldbody.add_light(type=mujoco.mjtLightType.mjLIGHT_POINT, diffuse=[1, 1, 1])
  expected = estimate_point_bounce(room)
  np.testing.assert_allclose(estimate_point_bounce(spec.compile()), expected, rtol=1e-5)
  prop = spec.worldbody.add_body(name="dynamic_prop")
  prop.add_freejoint()
  prop.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.3, 0, 0], rgba=[1, 0, 0, 1])
  np.testing.assert_allclose(estimate_point_bounce(spec.compile()), expected, rtol=1e-5)
