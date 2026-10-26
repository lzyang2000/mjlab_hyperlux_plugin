"""Move a G1 robot through a simple office with fixed sun and ceiling lights.

The room uses MJCF primitives; the robot comes from mjlab. This is a small
usage example, not the furnished SimReady scene in the published benchmarks.
"""

import json
import math
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

from mjlab_hyperlux import (
  HyperLuxCameraSensorCfg,
  apply_mujoco_lighting,
  get_lighting_profile,
)

OFFICE = """<mujoco model="windowed_office">
  <default><geom type="box" contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="tiles" type="2d" builtin="checker" width="128" height="128"
             rgb1=".38 .27 .16" rgb2=".48 .36 .23"/>
    <material name="floor" texture="tiles" texrepeat="6 5"/>
    <material name="wall" rgba=".72 .72 .68 1"/>
    <material name="back" rgba=".22 .38 .36 1"/>
    <material name="wood" rgba=".46 .32 .18 1"/>
    <material name="metal" rgba=".12 .14 .16 1"/>
  </asset>
  <worldbody>
    <light name="sun" type="directional" pos="-5 0 5" dir="1 .2 -.8"
           diffuse="2.4 2.15 1.8" ambient=".4 .43 .48"/>
    <light name="ceiling" type="point" pos="0 .1 2.85" diffuse="1.3 1.4 1.6"
           attenuation="1 0 .15"/>
    <geom name="floor" pos="0 0 -.05" size="3 2.6 .05" material="floor"/>
    <geom name="back_wall" pos="0 2.65 1.6" size="3.1 .1 1.6" material="back"/>
    <geom name="right_wall" pos="3.05 0 1.6" size=".1 2.6 1.6" material="wall"/>
    <geom name="front_wall" pos="0 -2.65 1.6" size="3.1 .1 1.6" material="wall"/>
    <geom name="ceiling" pos="0 0 3.25" size="3.1 2.7 .05" material="wall"/>
    <geom name="window_bottom" pos="-3.05 0 .45" size=".1 2.6 .45" material="wall"/>
    <geom name="window_top" pos="-3.05 0 2.95" size=".1 2.6 .25" material="wall"/>
    <geom name="window_side_a" pos="-3.05 -2.05 1.8" size=".1 .55 .9" material="wall"/>
    <geom name="window_side_b" pos="-3.05 2.05 1.8" size=".1 .55 .9" material="wall"/>
    <geom name="window_bar" pos="-3.05 0 1.8" size=".1 .04 .9" material="wall"/>
    <geom name="window_crossbar" pos="-3.05 0 1.8" size=".1 1.5 .04" material="wall"/>
    <geom name="desk" pos=".5 1.5 .75" size="1 .5 .05" material="wood"/>
    <geom name="desk_left" pos="-.3 1.5 .35" size=".06 .4 .35" material="metal"/>
    <geom name="desk_right" pos="1.3 1.5 .35" size=".06 .4 .35" material="metal"/>
    <geom name="monitor" pos=".5 1.75 1.1" size=".28 .04 .18" material="metal"/>
    <geom name="chair" pos=".5 .6 .45" size=".3 .3 .05" material="metal"/>
    <geom name="chair_back" pos=".5 .3 .75" size=".3 .04 .3" material="metal"/>
    <geom name="chair_base" pos=".5 .6 .2" size=".08 .08 .2" material="metal"/>
    <geom name="cabinet" pos="2 1.8 .8" size=".4 .45 .8" material="wall"/>
    <camera name="overview" pos="2.55 -2.25 2.35"
            xyaxes=".733 .680 0 -.215 .232 .949" fovy="68"/>
  </worldbody>
</mujoco>"""


def main(
  backend: Literal["hyperlux", "mujoco_warp"] = "hyperlux",
  num_envs: int = 1,
  width: int = 640,
  height: int = 480,
  batch_size: int = 128,
  output: Path = Path("outputs/windowed-office"),
) -> None:
  """Save a moving-robot GIF for one environment, or time a batch of environments."""
  if min(num_envs, width, height, batch_size) < 1:
    raise ValueError(
      "Environment count, image dimensions and batch size must be positive"
    )
  torch.cuda.set_device(0)
  camera = HyperLuxCameraSensorCfg(
    name="overview",
    camera_name="room/overview",
    width=width,
    height=height,
    rgb_backend=backend,
    use_shadows=True,
    hyperlux_anti_aliasing_grid=1,
    hyperlux_batch_size=batch_size,
    hyperlux_static_geoms=("room/*",),
    hyperlux_background=(0.45, 0.60, 0.78),
    enabled_geom_groups=(0, 1, 2),
  )
  scene = Scene(
    SceneCfg(
      num_envs=num_envs,
      entities={
        "room": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(OFFICE)),
        "robot": get_g1_robot_cfg(),
      },
      sensors=(camera,),
    ),
    "cuda:0",
  )
  model = scene.compile()
  model.vis.headlight.active = 0
  for light in range(model.nlight):
    model.light_active[light] = model.light(light).name.startswith("room/")
  apply_mujoco_lighting(model, get_lighting_profile("sun+indoor"), backend)
  sim = Simulation(num_envs, SimulationCfg(njmax=512, nconmax=128), model, "cuda:0")
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)
  try:
    robot = scene.entities["robot"]
    state = robot.data.default_root_state.clone()
    state[:, 1] = -0.3
    robot.write_joint_state_to_sim(
      robot.data.default_joint_pos, robot.data.default_joint_vel
    )

    def capture(frame: int) -> torch.Tensor:
      state[:, 0] = 0.45 * math.sin(2 * math.pi * frame / 24)
      robot.write_root_state_to_sim(state)
      sim.forward()
      sim.sense()
      return scene["overview"].data.rgb

    output.mkdir(parents=True, exist_ok=True)
    first = capture(0)[0].clone()
    Image.fromarray(first.cpu().numpy()).save(output / f"{backend}.png")
    for frame in range(3):
      capture(frame)
    torch.cuda.synchronize()
    samples = []
    for repeat in range(3):
      started = time.perf_counter()
      for frame in range(5):
        capture(6 + repeat * 5 + frame)
      torch.cuda.synchronize()
      samples.append((time.perf_counter() - started) * 1000 / 5)
    rgb = scene["overview"].data.rgb
    assert rgb.shape == (num_envs, height, width, 3)
    assert bool((rgb[:, ::16, ::16].amax(dim=(1, 2, 3)) > 0).all())
    assert (rgb[0].float() - first.float()).abs().mean().item() > 0.02
    result = {
      "scene": "procedural-windowed-office",
      "backend": backend,
      "aa_grid": 1 if backend == "hyperlux" else None,
      "num_envs": num_envs,
      "width": width,
      "height": height,
      "batch_size": batch_size if backend == "hyperlux" else None,
      "samples_ms": samples,
      "median_ms": statistics.median(samples),
      "images_per_second": num_envs * 1000 / statistics.median(samples),
      "gpu": torch.cuda.get_device_name(),
      "scope": "Root pose update, transforms and RGB GPU output; excludes physics, setup, CPU copies and encoding.",
    }
    (output / f"{backend}.json").write_text(json.dumps(result, indent=2) + "\n")
    if num_envs == 1:
      pictures = [
        Image.fromarray(capture(frame)[0].cpu().numpy().copy()) for frame in range(24)
      ]
      palette_source = Image.new("RGB", (160 * 24, 120))
      for frame, picture in enumerate(pictures):
        palette_source.paste(picture.resize((160, 120)), (160 * frame, 0))
      palette = palette_source.quantize(colors=256)
      frames = [picture.quantize(palette=palette) for picture in pictures]
      frames[0].save(
        output / f"{backend}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=False,
        disposal=1,
      )
    print(json.dumps(result, indent=2))
  finally:
    scene.sensor_context.close()


if __name__ == "__main__":
  tyro.cli(main)
