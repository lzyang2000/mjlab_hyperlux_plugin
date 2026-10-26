# mjlab-hyperlux

[HyperLux](https://pypi.org/project/hyperlux/) RGB rendering for
[mjlab](https://github.com/mujocolab/mjlab), with GPU output, moving objects
and hard shadows. MuJoCo Warp still handles physics, depth, segmentation
and raycasts.

In the original RTX 4090 measurements, HyperLux AA1 rendered our six scenes
**3.62x to 7.47x faster than Warp** at 4096 environments and 640x480. Results depend on
the scene and rendering settings; the measurements below exclude physics
and policy inference.

## Install

You need an NVIDIA GPU, a CUDA toolkit with `nvcc`, a C++ compiler, Git and uv.
This plugin requires the included mjlab RGB-interface patch.

```bash
git clone git@github.com:lzyang2000/mjlab_hyperlux_plugin.git
cd mjlab_hyperlux_plugin
```

Follow [the setup instructions](docs/setup.md) to install the tested mjlab
version and apply the mjlab and HyperLux patches. The first render compiles
native code; compatible settings reuse the cache across scenes.

## Use

```python
from mjlab_hyperlux import HyperLuxCameraSensorCfg

camera = HyperLuxCameraSensorCfg(
  name="front",
  camera_name="robot/front_camera",
  width=640,
  height=480,
  use_shadows=True,
  hyperlux_anti_aliasing_grid=1,
  hyperlux_batch_size="auto",
)
```

Add this camera to `SceneCfg.sensors`. RGB is a GPU `uint8` tensor shaped
`[num_envs, height, width, 3]`. The default render batch is 128; every environment
still receives an image. See [camera and lighting options](docs/usage.md).

After setup, run the small windowed-office example:

```bash
uv run --project mjlab --no-sync python examples/windowed_office.py
```

It saves a moving-robot GIF using built-in room geometry and mjlab's G1.
No SimReady download is needed. The furnished scenes pictured below are
saved comparisons, not bundled scene assets.

## Performance

RTX 4090, 4096 environments, one 640x480 RGB camera per environment. Cameras
and lights are fixed; robot roots move sideways with fixed joints. All
replicas use the same motion phase. HyperLux uses AA1 and batch 128; Warp
uses native RGB. The Warp timings below predate the blue-sky update and
used a black background. The warehouse and outdoor HyperLux/RTX measurements
also predate a box-texture fix. Those results are being refreshed.

Throughput is aggregate images per second. Speedup is HyperLux / Warp.

| Scene | HyperLux images/s | Warp images/s | Speedup | RTX images/s |
| --- | ---: | ---: | ---: | ---: |
| Windowed office | 14,050 | 1,881 | 7.47x | 32.6 |
| Home lounge | 12,858 | 3,549 | 3.62x | Not measured |
| Laboratory | 15,456 | 2,789 | 5.54x | Not measured |
| Warehouse A | 11,179 | 1,721 | 6.49x | Not measured |
| Warehouse B | 12,534 | 1,678 | 7.47x | Not measured |
| Outdoor loading yard | 20,747 | 5,364 | 3.87x | Not measured |

Sampled peak VRAM in GiB, including startup and the desktop:

| Scene | HyperLux | Warp | RTX |
| --- | ---: | ---: | ---: |
| Windowed office | 14.79 | 19.15 | 17.10 |
| Home lounge | 13.78 | 19.18 | Not measured |
| Laboratory | 14.49 | 19.54 | Not measured |
| Warehouse A | 8.19 | 13.46 | Not measured |
| Warehouse B | 8.19 | 13.11 | Not measured |
| Outdoor loading yard | 7.28 | 13.07 | Not measured |

Timings include pose updates, transforms, rendering and the full GPU RGB
output. They exclude setup, physics, CPU image copies and encoding. HyperLux
and Warp use the median of three repeats of five full batches after three
warmup batches. VRAM was sampled every 200 ms and may miss brief peaks.
[Saved measurements](docs/benchmarks/4096.json) include the raw timing samples.

The RTX measurement uses Isaac Lab 3.0 / Isaac Sim 6.2 with 16 cameras retargeted
across 4096 real scene replicas. It includes that scheduling cost and uses
three single-batch repeats after warmup. Retargeting limits temporal
accumulation, so these frames are noisier than the settled RTX GIFs. Only the
office has a validated RTX measurement. These compare the current integrations
at their stated settings, not identical image quality or maximum renderer speed.

## Image comparisons

Each table shows **HyperLux, MuJoCo Warp, then Isaac RTX**. The geometry, robot
pose, camera and light positions match. All images are 640x480. HyperLux uses
AA1 with hard shadows; RTX uses DLAA, four direct samples, eight ray bounces
and fixed exposure (f/5, 1/50 s, ISO 100). Light intensities are converted for
each backend. Sunlit RTX scenes also use a sky dome.

The ground and walls use the same source textures across backends, including
box-shaped surfaces. The same lighting calibration is shared across scenes. HyperLux's diffuse-fill
estimate does not reproduce RTX global illumination. Windowed rooms can look
brighter and flatter, and some warehouse areas darker. HyperLux and Warp use
a matching solid blue background in sunlit scenes; this adds no illumination.
The outdoor RTX view is overexposed at this indoor exposure.

The GIFs move only the robot, with the same 24-frame loop and fixed lights.
Playback is 10 fps, not a measure of renderer speed. GIF palette quantization
can introduce small color differences from the PNGs.

### Windowed office

| View | HyperLux | MuJoCo Warp | Isaac RTX |
| --- | --- | --- | --- |
| Overview | ![Windowed office: HyperLux, overview](docs/media/windowed_office/hyperlux-overview-shadows.png) | ![Windowed office: MuJoCo Warp, overview](docs/media/windowed_office/mujoco_warp-overview-shadows.png) | ![Windowed office: Isaac RTX, overview](docs/media/windowed_office/rtx-overview.png) |
| Moving robot | ![Windowed office: HyperLux, moving robot](docs/media/windowed_office/hyperlux-moving-robot.gif) | ![Windowed office: MuJoCo Warp, moving robot](docs/media/windowed_office/mujoco_warp-moving-robot.gif) | ![Windowed office: Isaac RTX, moving robot](docs/media/windowed_office/rtx-moving-robot.gif) |

### Home lounge

| View | HyperLux | MuJoCo Warp | Isaac RTX |
| --- | --- | --- | --- |
| Overview | ![Home lounge: HyperLux, overview](docs/media/home/hyperlux-overview-shadows.png) | ![Home lounge: MuJoCo Warp, overview](docs/media/home/mujoco_warp-overview-shadows.png) | ![Home lounge: Isaac RTX, overview](docs/media/home/rtx-overview.png) |
| Moving robot | ![Home lounge: HyperLux, moving robot](docs/media/home/hyperlux-moving-robot.gif) | ![Home lounge: MuJoCo Warp, moving robot](docs/media/home/mujoco_warp-moving-robot.gif) | ![Home lounge: Isaac RTX, moving robot](docs/media/home/rtx-moving-robot.gif) |

### Laboratory

| View | HyperLux | MuJoCo Warp | Isaac RTX |
| --- | --- | --- | --- |
| Overview | ![Laboratory: HyperLux, overview](docs/media/laboratory/hyperlux-overview-shadows.png) | ![Laboratory: MuJoCo Warp, overview](docs/media/laboratory/mujoco_warp-overview-shadows.png) | ![Laboratory: Isaac RTX, overview](docs/media/laboratory/rtx-overview.png) |
| Moving robot | ![Laboratory: HyperLux, moving robot](docs/media/laboratory/hyperlux-moving-robot.gif) | ![Laboratory: MuJoCo Warp, moving robot](docs/media/laboratory/mujoco_warp-moving-robot.gif) | ![Laboratory: Isaac RTX, moving robot](docs/media/laboratory/rtx-moving-robot.gif) |

### Warehouse A

| View | HyperLux | MuJoCo Warp | Isaac RTX |
| --- | --- | --- | --- |
| Overview | ![Warehouse A: HyperLux, overview](docs/media/windowless_warehouse_a/hyperlux-overview-shadows.png) | ![Warehouse A: MuJoCo Warp, overview](docs/media/windowless_warehouse_a/mujoco_warp-overview-shadows.png) | ![Warehouse A: Isaac RTX, overview](docs/media/windowless_warehouse_a/rtx-overview.png) |
| Moving robot | ![Warehouse A: HyperLux, moving robot](docs/media/windowless_warehouse_a/hyperlux-moving-robot.gif) | ![Warehouse A: MuJoCo Warp, moving robot](docs/media/windowless_warehouse_a/mujoco_warp-moving-robot.gif) | ![Warehouse A: Isaac RTX, moving robot](docs/media/windowless_warehouse_a/rtx-moving-robot.gif) |

### Warehouse B

| View | HyperLux | MuJoCo Warp | Isaac RTX |
| --- | --- | --- | --- |
| Overview | ![Warehouse B: HyperLux, overview](docs/media/windowless_warehouse_b/hyperlux-overview-shadows.png) | ![Warehouse B: MuJoCo Warp, overview](docs/media/windowless_warehouse_b/mujoco_warp-overview-shadows.png) | ![Warehouse B: Isaac RTX, overview](docs/media/windowless_warehouse_b/rtx-overview.png) |
| Moving robot | ![Warehouse B: HyperLux, moving robot](docs/media/windowless_warehouse_b/hyperlux-moving-robot.gif) | ![Warehouse B: MuJoCo Warp, moving robot](docs/media/windowless_warehouse_b/mujoco_warp-moving-robot.gif) | ![Warehouse B: Isaac RTX, moving robot](docs/media/windowless_warehouse_b/rtx-moving-robot.gif) |

### Outdoor loading yard

| View | HyperLux | MuJoCo Warp | Isaac RTX |
| --- | --- | --- | --- |
| Overview | ![Outdoor loading yard: HyperLux, overview](docs/media/outdoor/hyperlux-overview-shadows.png) | ![Outdoor loading yard: MuJoCo Warp, overview](docs/media/outdoor/mujoco_warp-overview-shadows.png) | ![Outdoor loading yard: Isaac RTX, overview](docs/media/outdoor/rtx-overview.png) |
| Moving robot | ![Outdoor loading yard: HyperLux, moving robot](docs/media/outdoor/hyperlux-moving-robot.gif) | ![Outdoor loading yard: MuJoCo Warp, moving robot](docs/media/outdoor/mujoco_warp-moving-robot.gif) | ![Outdoor loading yard: Isaac RTX, moving robot](docs/media/outdoor/rtx-moving-robot.gif) |

## Limits and license

Lights are fixed at initialization in this integration. Moving objects and
cameras are supported; per-frame light edits, soft shadows, transparent
materials, skyboxes and orthographic cameras are not.

Plugin code is [Apache-2.0](LICENSE). HyperLux is MIT licensed. The comparison
images depict NVIDIA SimReady assets under CC BY 4.0; see the
[asset attribution](docs/media/ATTRIBUTION.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). Scene assets and the experimental
Isaac Lab tooling are not included.
