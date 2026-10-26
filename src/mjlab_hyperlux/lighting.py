"""Renderer-neutral lighting profiles and one-time backend calibration."""

from dataclasses import dataclass
from typing import Literal, Mapping

import mujoco
import numpy as np

LightingComponent = Literal["sun", "indoor", "sky_ambient"]


@dataclass(frozen=True)
class RendererLightingScale:
  """Conversion from canonical light values to one renderer's light units."""

  sun: float = 1.0
  indoor: float = 1.0
  sky_ambient: float = 1.0
  point_attenuation: tuple[float, float, float] | None = None
  """Optional point-light falloff conversion; None preserves the MJCF values."""
  indoor_bounce_gain: float = 0.0
  """Scale a room-average bounce estimate; zero disables the approximation."""

  def for_component(self, component: LightingComponent) -> float:
    return getattr(self, component)


@dataclass(frozen=True)
class LightingProfile:
  """A reusable light composition with renderer-specific conversion factors."""

  name: str
  components: tuple[LightingComponent, ...]
  renderer_scales: Mapping[str, RendererLightingScale]

  def scale_for(self, backend: str, component: LightingComponent) -> float:
    try:
      calibration = self.renderer_scales[backend]
    except KeyError as exc:
      raise ValueError(f"No lighting calibration for backend {backend!r}") from exc
    return calibration.for_component(component)


# These are deliberately scene-independent. The values are one-time visual
# conversions to the RTX-like reference response used by this project.
LIGHTING_PROFILES: Mapping[str, LightingProfile] = {
  "sun": LightingProfile(
    name="sun",
    components=("sun",),
    renderer_scales={
      "hyperlux": RendererLightingScale(sun=1.5, sky_ambient=0.7),
      "mujoco_warp": RendererLightingScale(sun=1.0, sky_ambient=0.4),
      "isaac_rtx": RendererLightingScale(),
    },
  ),
  "indoor": LightingProfile(
    name="indoor",
    components=("indoor",),
    renderer_scales={
      "hyperlux": RendererLightingScale(
        indoor=3.8,
        sky_ambient=0.7,
        point_attenuation=(0.0, 0.0, 1.0),
        indoor_bounce_gain=0.55,
      ),
      "mujoco_warp": RendererLightingScale(indoor=1.3, sky_ambient=0.4),
      "isaac_rtx": RendererLightingScale(),
    },
  ),
  "sun+indoor": LightingProfile(
    name="sun+indoor",
    components=("sun", "indoor"),
    renderer_scales={
      "hyperlux": RendererLightingScale(
        sun=1.5,
        indoor=3.8,
        sky_ambient=0.7,
        point_attenuation=(0.0, 0.0, 1.0),
        indoor_bounce_gain=0.55,
      ),
      "mujoco_warp": RendererLightingScale(sun=1.0, indoor=1.3, sky_ambient=0.4),
      "isaac_rtx": RendererLightingScale(),
    },
  ),
}


def get_lighting_profile(name: str) -> LightingProfile:
  """Return a named generic profile, with a useful error for typos."""
  try:
    return LIGHTING_PROFILES[name]
  except KeyError as exc:
    choices = ", ".join(sorted(LIGHTING_PROFILES))
    raise ValueError(
      f"Unknown lighting profile {name!r}; choose from {choices}"
    ) from exc


def apply_mujoco_lighting(
  model: mujoco.MjModel,
  profile: LightingProfile,
  backend: str,
) -> None:
  """Apply a profile's one-time calibration to a compiled MuJoCo model."""
  scales = profile.renderer_scales.get(backend)
  if scales is None:
    raise ValueError(f"No lighting calibration for backend {backend!r}")
  model.light_ambient[:] *= scales.sky_ambient
  for light_id in range(model.nlight):
    if model.light_type[light_id] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
      model.light_diffuse[light_id] *= scales.sun
    else:
      model.light_diffuse[light_id] *= scales.indoor
      if (
        scales.point_attenuation is not None
        and model.light_type[light_id] == mujoco.mjtLightType.mjLIGHT_POINT
      ):
        model.light_attenuation[light_id] = scales.point_attenuation
  if scales.indoor_bounce_gain > 0:
    points = np.flatnonzero(
      model.light_active & (model.light_type == mujoco.mjtLightType.mjLIGHT_POINT)
    )
    if len(points):
      existing = model.light_ambient[model.light_active != 0].sum(axis=0)
      # Authored ambient already approximates bounced light. Add only a deficit.
      model.light_ambient[points[0]] += np.maximum(
        estimate_point_bounce(model) * scales.indoor_bounce_gain - existing, 0.0
      )


def estimate_point_bounce(model: mujoco.MjModel) -> np.ndarray:
  """Estimate uniform diffuse fill for a static room with inverse-square lights.

  A deterministic sphere of rays estimates visible area, reflectance and escape
  from each point light. A well-mixed diffuse enclosure gives 4 I rho / A for
  the first bounce, with a geometric series for subsequent bounces. This is a
  room-average approximation, not spatial GI: disconnected rooms should not
  share this fill. Articulated geometry does not contribute to the estimate;
  mocap room roots are sampled at their authored pose.
  """
  import copy

  ray_model = copy.copy(model)
  # mjlab wraps fixed scene entities in mocap roots, which have nonzero weld
  # IDs. Exclude joint descendants, rather than every non-world weld group.
  articulated = np.zeros(ray_model.nbody, dtype=bool)
  for body in range(1, ray_model.nbody):
    articulated[body] = (
      articulated[ray_model.body_parentid[body]] or ray_model.body_dofnum[body] > 0
    )
  ray_model.geom_group[articulated[ray_model.geom_bodyid]] = 5
  data = mujoco.MjData(ray_model)
  mujoco.mj_forward(ray_model, data)
  count = 512
  indices = np.arange(count)
  z = 1 - 2 * (indices + 0.5) / count
  phi = indices * (np.pi * (3 - np.sqrt(5)))
  directions = np.column_stack(
    (
      np.sqrt(1 - z * z) * np.cos(phi),
      np.sqrt(1 - z * z) * np.sin(phi),
      z,
    )
  )
  groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
  ids = np.empty(count, dtype=np.int32)
  distances = np.empty(count)
  normals = np.empty((count, 3))
  reflectance = np.zeros((model.ngeom, 3))
  texture_means = {}
  for geom in range(model.ngeom):
    mid = model.geom_matid[geom]
    color = (
      model.mat_rgba[mid, :3].copy() if mid >= 0 else model.geom_rgba[geom, :3].copy()
    )
    metallic = max(0.0, float(model.mat_metallic[mid])) if mid >= 0 else 0.0
    if mid >= 0:
      for role in (
        mujoco.mjtTextureRole.mjTEXROLE_RGB,
        mujoco.mjtTextureRole.mjTEXROLE_ORM,
      ):
        tid = int(model.mat_texid[mid, role])
        if tid < 0:
          continue
        key = (tid, int(role))
        if key not in texture_means:
          h, w, c = model.tex_height[tid], model.tex_width[tid], model.tex_nchannel[tid]
          pixels = (
            model.tex_data[model.tex_adr[tid] : model.tex_adr[tid] + h * w * c].reshape(
              h, w, c
            )[..., :3]
            / 255.0
          )
          if c == 1:
            pixels = np.repeat(pixels, 3, axis=2)
          if role == mujoco.mjtTextureRole.mjTEXROLE_RGB:
            pixels = np.where(
              pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4
            )
          texture_means[key] = pixels.mean(axis=(0, 1))
        if role == mujoco.mjtTextureRole.mjTEXROLE_RGB:
          color *= texture_means[key]
        else:
          metallic *= texture_means[key][2]
    reflectance[geom] = np.clip(color * (1 - metallic), 0, 0.9)
  ambient = np.zeros(3)
  for light in range(model.nlight):
    if (
      not model.light_active[light]
      or model.light_type[light] != mujoco.mjtLightType.mjLIGHT_POINT
    ):
      continue
    mujoco.mj_multiRay(
      ray_model,
      data,
      data.light_xpos[light],
      directions.ravel(),
      groups,
      True,
      -1,
      ids,
      distances,
      normals.ravel(),
      count,
      50.0,
    )
    hit = (ids >= 0) & (distances > 1e-4)
    if not hit.any():
      continue
    cosine = np.maximum(np.abs(np.sum(normals[hit] * directions[hit], axis=1)), 1e-4)
    weights = distances[hit] ** 2 / cosine
    area = 4 * np.pi * weights.sum() / count
    # Escape reduces the energy retained on later diffuse bounces.
    enclosure = hit.mean()
    rho = (reflectance[ids[hit]] * weights[:, None]).sum(axis=0) / weights.sum()
    ambient += (
      4 * model.light_diffuse[light] * enclosure * rho / (area * (1 - enclosure * rho))
    )
  return ambient
