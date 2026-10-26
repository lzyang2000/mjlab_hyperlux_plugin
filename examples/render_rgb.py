"""Compare RGB backends on identical G1 scenes; save images and wall-clock timings.

Example: uv run examples/render_rgb.py --backend hyperlux --num-envs 32
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Literal

import mujoco
import torch
import tyro
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from PIL import Image

from mjlab_hyperlux import HyperLuxCameraSensorCfg

WORLD = """<mujoco>
  <visual><headlight active="0"/></visual>
  <asset>
    <texture name="tiles" type="2d" builtin="checker" width="128" height="128"
             rgb1=".15 .2 .25" rgb2=".65 .7 .75"/>
    <material name="floor" texture="tiles" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light name="sun" type="directional" pos="0 -2 4" dir="0 .5 -1"
           ambient=".25 .25 .25" diffuse=".8 .8 .8"/>
    <geom type="plane" size="6 6 .1" material="floor"/>
    <camera name="cam" pos="2.5 -2.5 1.8" xyaxes=".707 .707 0 -.236 .236 .943"
            fovy="45"/>
  </worldbody>
</mujoco>"""


def main(
  backend: Literal["mujoco_warp", "hyperlux"] = "hyperlux",
  num_envs: int = 32,
  width: int = 160,
  height: int = 120,
  shadows: bool = False,
  aa: Literal[1, 2, 3, 4] = 1,
  batch_size: int = 128,
  steps: int = 100,
  warmup: int = 20,
  repeats: int = 3,
  device: str = "cuda:0",
  output: Path = Path("benchmark_results/hyperlux"),
  light_direction: tuple[float, float, float] = (0.0, 0.5, -1.0),
  ambient: float = 0.25,
) -> None:
  torch.cuda.set_device(device)
  free_before, total_memory = torch.cuda.mem_get_info(device)
  print(f"Building {backend}: {num_envs} cameras at {width}x{height}", flush=True)
  camera = HyperLuxCameraSensorCfg(
    name="camera",
    camera_name="world/cam",
    width=width,
    height=height,
    rgb_backend=backend,
    use_shadows=shadows,
    hyperlux_anti_aliasing_grid=aa,
    hyperlux_batch_size=batch_size,
    enabled_geom_groups=(0, 1, 2),
  )
  world_xml = WORLD.replace(
    'dir="0 .5 -1"',
    f'dir="{light_direction[0]} {light_direction[1]} {light_direction[2]}"',
  ).replace('ambient=".25 .25 .25"', f'ambient="{ambient} {ambient} {ambient}"')
  scene = Scene(
    SceneCfg(
      num_envs=num_envs,
      entities={
        "robot": get_g1_robot_cfg(),
        "world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(world_xml)),
      },
      sensors=(camera,),
    ),
    device,
  )
  mj_model = scene.compile()
  # Entity XML visual settings do not replace mjlab's scene defaults; G1 also
  # brings its own light. Use exactly the benchmark's authored directional light.
  mj_model.vis.headlight.active = 0
  mj_model.light_active[:] = False
  mj_model.light_active[mj_model.light("world/sun").id] = True
  sim = Simulation(num_envs, SimulationCfg(njmax=512, nconmax=128), mj_model, device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)
  robot = scene.entities["robot"]

  def reset() -> None:
    sim.reset()
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(
      robot.data.default_joint_pos, robot.data.default_joint_vel
    )
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.write_data_to_sim()
    sim.forward()

  reset()
  print("Rendering preview", flush=True)
  sim.sense()
  rgb = scene["camera"].data.rgb
  assert rgb.shape == (num_envs, height, width, 3)
  assert bool((rgb.amax(dim=(1, 2, 3)) > 0).all())
  assert torch.equal(rgb[0], rgb[-1])  # Identical initial states, including last chunk.
  output.mkdir(parents=True, exist_ok=True)
  name = f"{backend}-{num_envs}-{width}x{height}-shadows{int(shadows)}-aa{aa}"
  Image.fromarray(scene["camera"].data.rgb[0].cpu().numpy()).save(
    output / f"{name}.png"
  )
  print("Preview saved; timing", flush=True)
  free_setup, _ = torch.cuda.mem_get_info(device)

  def run(physics: bool) -> None:
    if physics:
      sim.step()
      # Camera/geom world poses must correspond to the integrated state.
      sim.forward()
    sim.sense()

  timings = {}
  for mode, physics in (("sense", False), ("step_forward_sense", True)):
    samples = []
    for _ in range(repeats):
      reset()
      for _ in range(warmup):
        run(physics)
      torch.cuda.synchronize(device)
      start = time.perf_counter()
      for _ in range(steps):
        run(physics)
      torch.cuda.synchronize(device)
      samples.append((time.perf_counter() - start) * 1000 / steps)
    median = statistics.median(samples)
    timings[mode] = {
      "batch_ms": median,
      "images_per_second": num_envs * 1000 / median,
      "samples_ms": samples,
    }
  result = {
    "backend": backend,
    "num_envs": num_envs,
    "width": width,
    "height": height,
    "shadows": shadows,
    "light_direction": light_direction,
    "ambient": ambient,
    "aa": aa,
    "steps": steps,
    "warmup": warmup,
    "hyperlux_batch_size": batch_size,
    "rgb_shape": list(rgb.shape),
    "gpu": torch.cuda.get_device_name(device),
    "timings": timings,
    "torch": torch.__version__,
    "mujoco": mujoco.__version__,
    "peak_torch_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
    "device_used_gib_before_setup": (total_memory - free_before) / 2**30,
    "device_used_gib_after_setup": (total_memory - free_setup) / 2**30,
    "device_used_gib_after_benchmark": (
      total_memory - torch.cuda.mem_get_info(device)[0]
    )
    / 2**30,
  }
  (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  scene.sensor_context.close()


if __name__ == "__main__":
  tyro.cli(main)
