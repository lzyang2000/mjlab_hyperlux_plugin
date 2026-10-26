# Camera options

Add `HyperLuxCameraSensorCfg` to `SceneCfg.sensors` as you would an mjlab camera.
RGB output is a GPU `uint8` tensor with shape `[num_envs, height, width, 3]`.
Physics, depth, segmentation and raycasts use MuJoCo Warp.

```python
from mjlab_hyperlux import HyperLuxCameraSensorCfg

camera = HyperLuxCameraSensorCfg(
  name="front",
  camera_name="robot/front_camera",
  width=640,
  height=480,
  data_types=("rgb",),
  use_shadows=True,
  hyperlux_anti_aliasing_grid=1,
  hyperlux_batch_size="auto",
)
```

AA1 is one sample per pixel. AA2 uses a 2x2 grid, so it costs substantially more.
`hyperlux_batch_size="auto"` selects 128 environments per submission. It does
not reduce the number of simulated environments or the full RGB output tensor.
At 4096 environments and 640x480, that tensor alone is about 3.52 GiB.

Larger batches did not improve throughput in our 4090 sweep and used more
memory. A single 4096-camera launch failed, so the default submits smaller
batches. You can set an integer batch size for your workload.

## Static and moving geometry

```python
camera = HyperLuxCameraSensorCfg(
  name="front",
  camera_name="robot/front_camera",
  hyperlux_static_geoms=("world/floor", "world/wall/*"),
  hyperlux_dynamic_geoms=("terrain/*", "obstacle/*"),
)
```

These names or glob patterns match compiled MuJoCo geom names. Static geometry
is imported once and shared. It must have the same initial world pose in every
environment and remain fixed. Put randomized terrain and moving objects in
the dynamic set. Overlapping labels or differing initial static poses raise
an error. Unmatched geometry remains dynamic.

Each environment has its own dynamic overlay. Robots in separate environments
do not appear in each other's images. Multiple robots in the same environment
can see, occlude and cast shadows on one another. Geoms on the same body are
packed into one asset to reduce pose-update work.

## Lighting and supported features

The backend supports perspective cameras, 2D textures on meshes, planes and
boxes, normal and packed ORM
maps, moving cameras and objects, and hard shadows from directional, point
and spot lights. Set `hyperlux_background` for the background RGB color.

`hyperlux_fidelity` accepts `low`, `medium`, `high` or `veryhigh`. The default
selects `veryhigh` with shadows and `high` without them.

Optional `sun`, `indoor` and `sun+indoor` profiles convert light intensities
for the different backends. Apply one to the compiled MuJoCo model before
initializing the renderer:

```python
from mjlab_hyperlux import apply_mujoco_lighting, get_lighting_profile

apply_mujoco_lighting(model, get_lighting_profile("sun+indoor"), "hyperlux")
```

These are shared visual calibrations, not physical photometric units. HyperLux
uses inverse-square point-light attenuation and an optional room-average
diffuse-fill estimate computed during setup. This does not reproduce RTX's
spatial global illumination.

This integration does not expose per-frame light changes, soft shadows,
transparent materials, skyboxes, cubemaps or orthographic cameras. MuJoCo,
HyperLux and RTX use different material and lighting models.
