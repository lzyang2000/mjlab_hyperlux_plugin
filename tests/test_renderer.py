"""Exercise the actual HyperLux backend through mjlab's sensing pipeline."""

import mujoco
import pytest
import torch

pytest.importorskip("hyperlux")

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import Simulation, SimulationCfg

from mjlab_hyperlux import HyperLuxCameraSensorCfg

from .conftest import _make_scene_and_sim, get_test_device

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="HyperLux integration tests require CUDA/OptiX"
)
XML = """<mujoco>
  <visual><headlight ambient="0.5 0.5 0.5"/></visual>
  <worldbody>
    <light type="directional" pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="4 4 .1" rgba=".3 .3 .3 1"/>
    <body name="box" pos="0 0 .5">
      <freejoint/>
      <geom name="red" type="box" size=".3 .2 .5" rgba="1 0 0 1"/>
    </body>
    <camera name="cam" pos="0 0 3" quat="1 0 0 0" fovy="60"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def setup():
  cfg = HyperLuxCameraSensorCfg(
    name="camera",
    camera_name="world/cam",
    width=32,
    height=24,
    rgb_backend="hyperlux",
    data_types=("rgb", "depth", "segmentation"),
  )
  scene, sim = _make_scene_and_sim(get_test_device(), (cfg,), xml=XML)
  sim.forward()
  sim.sense()
  yield scene, sim
  assert scene.sensor_context is not None
  scene.sensor_context.close()


def test_rgb_matches_depth_and_segmentation(setup):
  scene, sim = setup
  output = scene["camera"].data
  assert output.rgb.shape == (2, 24, 32, 3)
  assert output.rgb.dtype == torch.uint8
  assert output.rgb.device.type == get_test_device().split(":")[0]
  rgb = output.rgb.to(torch.int32)
  red = (rgb[..., 0] > rgb[..., 1] * 2) & (rgb[..., 0] > 30)
  geom = sim.mj_model.geom("world/red").id
  mask = output.segmentation[..., 0] == geom
  assert mask.any()
  assert (red == mask).float().mean() > 0.98
  assert torch.allclose(
    output.depth[mask], torch.full_like(output.depth[mask], 2.0), atol=0.01
  )


def test_movement_isolated_and_reset(setup):
  scene, sim = setup
  original = scene["camera"].data.rgb.clone()
  sim.data.qpos[0, 0] = 0.8
  sim.forward()
  sim.sense()
  moved = scene["camera"].data.rgb.clone()
  assert not torch.equal(moved[0], original[0])
  assert torch.equal(moved[1], original[1])
  sim.reset(torch.tensor([0], device=get_test_device()))
  sim.forward()
  sim.sense()
  assert torch.equal(scene["camera"].data.rgb, original)


def test_rgb_only_skips_warp_rendering():
  cfg = HyperLuxCameraSensorCfg(
    name="camera",
    camera_name="world/cam",
    width=32,
    height=24,
    rgb_backend="hyperlux",
  )
  scene, sim = _make_scene_and_sim(get_test_device(), (cfg,), xml=XML)
  sim.forward()
  sim.sense()
  assert scene.sensor_context is not None
  assert not scene.sensor_context.has_warp_sensors
  assert sim.sense_graph is None
  assert scene["camera"].data.rgb.any()
  scene.sensor_context.close()


def test_two_cameras_and_backends():
  sensors = tuple(
    HyperLuxCameraSensorCfg(
      name=f"cam{i}",
      pos=(x, 0.0, 3.0),
      width=32,
      height=24,
      rgb_backend="hyperlux" if i < 2 else "mujoco_warp",
    )
    for i, x in enumerate([0.0, 0.8, 0.0])
  )
  scene, sim = _make_scene_and_sim(get_test_device(), sensors, xml=XML)
  sim.forward()
  sim.sense()
  a, b, c = (scene[f"cam{i}"].data.rgb for i in range(3))
  assert a.shape == b.shape == c.shape
  assert a.any() and b.any() and c.any()
  assert not torch.equal(a, b)
  assert scene.sensor_context is not None
  scene.sensor_context.close()


@pytest.mark.parametrize("field", ["cam_fovy", "cam_intrinsic"])
def test_per_world_projection_and_camera_motion(field):
  xml = XML
  if field == "cam_intrinsic":
    xml = xml.replace(
      'fovy="60"',
      'sensorsize=".032 .024" focal=".02 .02" '
      'principal=".002 -.001" resolution="32 24"',
    )
  cfg = HyperLuxCameraSensorCfg(
    name="camera",
    camera_name="world/cam",
    width=32,
    height=24,
    rgb_backend="hyperlux",
    data_types=("rgb", "segmentation"),
  )
  scene, sim = _make_scene_and_sim(get_test_device(), (cfg,), xml=xml)
  sim.expand_model_fields((field, "cam_pos"))
  sim.forward()
  sim.sense()
  original = scene["camera"].data.rgb.clone()
  cam = sim.mj_model.camera("world/cam").id
  if field == "cam_fovy":
    sim.model.cam_fovy[0, cam] = 35.0
  else:
    sim.model.cam_intrinsic[0, cam, :2] *= 1.5
  sim.model.cam_pos[0, cam, 0] = 0.15
  sim.forward()
  sim.sense()
  output = scene["camera"].data
  assert not torch.equal(output.rgb[0], original[0])
  assert torch.equal(output.rgb[1], original[1])
  red = output.rgb[..., 0].int() > output.rgb[..., 1].int() * 2
  mask = output.segmentation[..., 0] == sim.mj_model.geom("world/red").id
  assert (red == mask).float().mean() > 0.98
  assert scene.sensor_context is not None
  scene.sensor_context.close()


def test_rgb_with_raycast():
  camera = HyperLuxCameraSensorCfg(
    name="camera",
    camera_name="world/cam",
    width=32,
    height=24,
    rgb_backend="hyperlux",
  )
  raycast = RayCastSensorCfg(
    name="scan",
    frame=ObjRef(type="body", name="box", entity="world"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1),
    exclude_parent_body=True,
    include_geom_groups=(0, 1, 2),
  )
  scene, sim = _make_scene_and_sim(get_test_device(), (camera, raycast), xml=XML)
  sim.forward()
  sim.sense()
  assert scene["camera"].data.rgb.any()
  distances = scene["scan"].data.distances
  assert torch.allclose(distances, torch.full_like(distances, 0.5), atol=0.001)
  assert scene.sensor_context is not None
  assert not scene.sensor_context.has_warp_cameras
  assert scene.sensor_context.has_warp_sensors
  scene.sensor_context.close()


@pytest.mark.parametrize("floor_type", ["plane", "box"])
def test_textured_primitives_and_separate_camera_batches(floor_type):
  xml = XML.replace(
    "<worldbody>",
    """<asset>
    <texture name="tiles" type="2d" builtin="checker" width="32" height="32"
      rgb1="0 1 0" rgb2="0 0 1"/>
    <material name="floor_mat" texture="tiles" texrepeat="4 4"/>
  </asset><worldbody>""",
  ).replace('size="4 4 .1" rgba=".3 .3 .3 1"', 'size="0 0 .1" material="floor_mat"')
  if floor_type == "box":
    xml = xml.replace('type="plane" size="0 0 .1"', 'type="box" size="4 4 .1"')
  cameras = (
    HyperLuxCameraSensorCfg(name="a", camera_name="world/cam", width=32, height=24),
    HyperLuxCameraSensorCfg(
      name="b", pos=(0.0, 0.0, 3.0), width=48, height=32, hyperlux_anti_aliasing_grid=2
    ),
  )
  scene, sim = _make_scene_and_sim(get_test_device(), cameras, xml=xml)
  sim.forward()
  sim.sense()
  assert scene["a"].data.rgb.shape == (2, 24, 32, 3)
  assert scene["b"].data.rgb.shape == (2, 32, 48, 3)
  for name in ("a", "b"):
    rgb = scene[name].data.rgb.int()
    assert (rgb[..., 1] > rgb[..., 0] + 30).any()
    assert (rgb[..., 2] > rgb[..., 0] + 30).any()
  assert scene.sensor_context is not None
  scene.sensor_context.close()


@pytest.mark.parametrize(
  "light_type,metallic",
  [("directional", False), ("point", False), ("spot", False), ("directional", True)],
)
def test_shadows_move_and_stay_in_their_world(light_type, metallic):
  xml = XML.replace('ambient="0.5 0.5 0.5"', 'ambient="0.05 0.05 0.05"')
  xml = xml.replace(
    '<light type="directional" pos="0 0 4" dir="0 0 -1"/>',
    f'<light type="{light_type}" pos="-4 0 4" dir="1 0 -1" '
    'diffuse="1 1 1" attenuation="1 0 0" cutoff="70"/>',
  ).replace('rgba=".3 .3 .3 1"', 'rgba=".7 .7 .7 1"')
  if metallic:
    xml = xml.replace(
      "<worldbody>",
      '<asset><material name="metal" metallic="0.8" roughness="0.15"/>'
      '<texture name="tiles" type="2d" builtin="checker" width="32" height="32" '
      'rgb1=".7 .7 .7" rgb2=".5 .5 .5"/>'
      '<material name="tiles" texture="tiles" texrepeat="4 4"/></asset><worldbody>',
    )
    xml = xml.replace('name="red"', 'name="red" material="metal"')
    xml = xml.replace('name="floor"', 'name="floor" material="tiles"')
  images = {}
  floors = {}
  for shadows in (False, True):
    cfg = HyperLuxCameraSensorCfg(
      name="camera",
      camera_name="world/cam",
      width=64,
      height=48,
      use_shadows=shadows,
      hyperlux_batch_size=2,
      data_types=("rgb", "segmentation"),
    )
    scene, sim = _make_scene_and_sim(get_test_device(), (cfg,), xml=xml, num_envs=5)
    floor_id = sim.mj_model.geom("world/floor").id
    for moved in (False, True):
      if moved:
        sim.data.qpos[0, 0] = -0.7
        sim.data.qpos[4, 0] = 0.5
      sim.forward()
      sim.sense()
      output = scene["camera"].data
      images[shadows, moved] = output.rgb.clone()
      floors[moved] = output.segmentation[..., 0] == floor_id
    assert scene.sensor_context is not None
    scene.sensor_context.close()

  masks = []
  for moved in (False, True):
    darkening = (images[False, moved].float() - images[True, moved].float()).mean(
      dim=-1
    )
    shadow = (darkening > 15) & floors[moved]
    assert (shadow.sum(dim=(1, 2)) > 10).all()
    masks.append(shadow)
  # Independent worlds, including the last, partially filled chunk.
  assert torch.equal(images[True, False][1:4], images[True, True][1:4])
  for world, direction in ((0, -1), (4, 1)):
    before_x = masks[0][world].nonzero()[:, 1].float().mean()
    after_x = masks[1][world].nonzero()[:, 1].float().mean()
    assert direction * (after_x - before_x) > 3


def test_chunked_worlds_and_partial_last_batch():
  cfg = HyperLuxCameraSensorCfg(
    name="camera",
    camera_name="world/cam",
    width=32,
    height=24,
    hyperlux_batch_size=2,
    data_types=("rgb", "segmentation"),
  )
  scene, sim = _make_scene_and_sim(get_test_device(), (cfg,), xml=XML, num_envs=5)
  sim.data.qpos[:, 0] = torch.linspace(-0.6, 0.6, 5, device=get_test_device())
  sim.forward()
  sim.sense()
  data = scene["camera"].data
  red = data.rgb[..., 0].int() > data.rgb[..., 1].int() * 2
  mask = data.segmentation[..., 0] == sim.mj_model.geom("world/red").id
  assert (red == mask).float().mean() > 0.98
  assert not torch.equal(data.rgb[0], data.rgb[4])
  saved = data.rgb.clone()
  sim.sense()
  assert torch.equal(saved, scene["camera"].data.rgb)
  assert scene.sensor_context is not None
  scene.sensor_context.close()


def test_g1_casts_a_ground_shadow_beyond_its_silhouette():
  xml = """<mujoco><asset>
    <texture name="tiles" type="2d" builtin="checker" width="32" height="32"
      rgb1=".5 .5 .5" rgb2=".8 .8 .8"/>
    <material name="tiles" texture="tiles" texrepeat="8 8"/>
  </asset><worldbody>
    <light name="sun" type="directional" dir="1 1 -1"
      ambient=".05 .05 .05" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="6 6 .1" material="tiles"/>
    <camera name="cam" pos="2.5 -2.5 1.8" xyaxes=".707 .707 0 -.236 .236 .943"/>
  </worldbody></mujoco>"""
  frames = {}
  floors = {}
  for shadows in (False, True):
    cfg = HyperLuxCameraSensorCfg(
      name="camera",
      camera_name="world/cam",
      width=160,
      height=120,
      use_shadows=shadows,
      hyperlux_batch_size=1,
      data_types=("rgb", "segmentation"),
    )
    scene = Scene(
      SceneCfg(
        num_envs=2,
        entities={
          "robot": get_g1_robot_cfg(),
          "world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml)),
        },
        sensors=(cfg,),
      ),
      get_test_device(),
    )
    model = scene.compile()
    model.vis.headlight.active = 0
    model.light_active[:] = False
    model.light_active[model.light("world/sun").id] = True
    sim = Simulation(2, SimulationCfg(njmax=512, nconmax=128), model, get_test_device())
    scene.initialize(sim.mj_model, sim.model, sim.data)
    assert scene.sensor_context is not None
    sim.set_sensor_context(scene.sensor_context)
    robot = scene.entities["robot"]
    state = robot.data.default_root_state.clone()
    for moved in (False, True):
      if moved:
        state[0, 0] += 0.4
      robot.write_root_state_to_sim(state)
      robot.write_joint_state_to_sim(
        robot.data.default_joint_pos, robot.data.default_joint_vel
      )
      sim.forward()
      sim.sense()
      output = scene["camera"].data
      frames[shadows, moved] = output.rgb.clone()
      floors[moved] = output.segmentation[..., 0] == model.geom("world/floor").id
    scene.sensor_context.close()

  for moved in (False, True):
    delta = (frames[False, moved].float() - frames[True, moved].float()).mean(-1)
    # This region is on the floor, well to the right of the robot. Self-shading
    # or a dark robot surface cannot satisfy this regression assertion.
    projected = (delta > 8) & floors[moved]
    assert (projected[:, :, 105:].sum(dim=(1, 2)) > 20).all()
  assert not torch.equal(frames[True, False][0], frames[True, True][0])
  assert torch.equal(frames[True, False][1], frames[True, True][1])


def test_normal_and_packed_orm_maps_change_the_rendered_surface():
  frames = {}
  for name, normal, orm in (
    ("matte", ".5 .5 1", "1 1 0"),
    ("metal", ".5 .5 1", "1 1 1"),
    ("tilted", "1 .5 .5", "1 1 0"),
  ):
    xml = f"""<mujoco><asset>
      <texture name="color" type="2d" builtin="flat" width="4" height="4" rgb1="1 1 1"/>
      <texture name="normal" type="2d" builtin="flat" width="4" height="4" rgb1="{normal}"/>
      <texture name="orm" type="2d" builtin="flat" width="4" height="4" rgb1="{orm}"/>
      <material name="surface" metallic="1" roughness="1">
        <layer role="rgb" texture="color"/><layer role="normal" texture="normal"/>
        <layer role="orm" texture="orm"/>
      </material>
    </asset><worldbody>
      <light type="directional" dir="0 0 -1" diffuse="1 1 1"/>
      <geom type="plane" size="4 4 .1" material="surface"/>
      <camera name="cam" pos="0 0 3"/>
    </worldbody></mujoco>"""
    cfg = HyperLuxCameraSensorCfg(
      name="camera", camera_name="world/cam", width=32, height=24
    )
    scene, sim = _make_scene_and_sim(get_test_device(), (cfg,), xml=xml)
    sim.forward()
    sim.sense()
    frames[name] = scene["camera"].data.rgb.float().mean().item()
    assert scene.sensor_context is not None
    scene.sensor_context.close()
  # A missing ORM image would treat all three materials as fully metallic.
  assert frames["matte"] > frames["metal"] + 30
  assert frames["matte"] > frames["tilted"] + 20


def test_camera_backgrounds_have_independent_render_groups():
  sensors = tuple(
    HyperLuxCameraSensorCfg(
      name=name,
      pos=(0, 0, 3),
      quat=(0, 1, 0, 0),
      width=32,
      height=24,
      hyperlux_background=color,
    )
    for name, color in (("red", (1.0, 0.0, 0.0)), ("blue", (0.0, 0.0, 1.0)))
  )
  scene, sim = _make_scene_and_sim(get_test_device(), sensors, xml=XML)
  sim.forward()
  sim.sense()
  assert (
    scene["red"].data.rgb == torch.tensor([255, 0, 0], device=get_test_device())
  ).all()
  assert (
    scene["blue"].data.rgb == torch.tensor([0, 0, 255], device=get_test_device())
  ).all()
  assert scene.sensor_context is not None
  scene.sensor_context.close()


def test_box_wall_has_flat_normals_under_directional_light():
  xml = """<mujoco><worldbody>
    <light type="directional" dir=".3 1 -.4" diffuse="1 1 1"/>
    <geom type="box" size="2 .1 2" rgba=".5 .5 .5 1"/>
    <camera name="cam" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="32"/>
  </worldbody></mujoco>"""
  cfg = HyperLuxCameraSensorCfg(
    name="camera", camera_name="world/cam", width=64, height=48
  )
  scene, sim = _make_scene_and_sim(get_test_device(), (cfg,), xml=xml)
  sim.forward()
  sim.sense()
  rgb = scene["camera"].data.rgb.float()
  assert rgb.mean() > 10
  # All pixels hit one planar box face with the same normal and incident light.
  assert (rgb.amax(dim=(1, 2)) - rgb.amin(dim=(1, 2))).max() <= 1
  assert scene.sensor_context is not None
  scene.sensor_context.close()
