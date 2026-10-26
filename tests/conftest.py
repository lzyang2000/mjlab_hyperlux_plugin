"""Build real mjlab scenes for plugin integration tests."""

import mujoco
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg

wp.config.log_level = wp.LOG_WARNING


def get_test_device() -> str:
  return "cuda" if torch.cuda.is_available() else "cpu"


def _make_scene_and_sim(device: str, sensors: tuple, xml: str, num_envs: int = 2):
  scene = Scene(
    SceneCfg(
      num_envs=num_envs,
      entities={"world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))},
      sensors=sensors,
    ),
    device,
  )
  sim = Simulation(num_envs, SimulationCfg(), scene.compile(), device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)
  return scene, sim
