"""HyperLux RGB cameras driven by MuJoCo's computed geometry transforms.

User-labeled static meshes live in one shared scene. Dynamic meshes are placed
in per-world overlays attached only to that world's cameras. Physics and camera
kinematics remain in MuJoCo Warp.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from fnmatch import fnmatchcase
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import hyperlux
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import trimesh
import trimesh.visual.material
import trimesh.visual.texture
import warp as wp
from hyperlux.camera import RGB, CameraSpec
from mjlab.sensor.camera_sensor import CameraSensor
from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh
from PIL import Image
from torch.utils.dlpack import from_dlpack
from warp._src.context import Device

from mjlab_hyperlux.config import HyperLuxCameraSensorCfg


@wp.kernel
def _pack_poses(
  ids: wp.array(dtype=int),  # type: ignore[valid-type]
  xpos: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
  xmat: wp.array2d(dtype=wp.mat33),  # type: ignore[valid-type]
  world_offset: int,
  poses: wp.array4d(dtype=float),  # type: ignore[valid-type]
):
  world, index = wp.tid()  # type: ignore[attr-defined]
  source = (world_offset + world) % xpos.shape[0]  # Pad the last batch.
  body = ids[index]
  rotation = xmat[source, body]
  position = xpos[source, body]
  for row in range(3):
    for col in range(3):
      poses[world, index, row, col] = rotation[row, col]
    poses[world, index, row, 3] = position[row]


@wp.kernel
def _unpack_rgb_packed(
  packed: wp.array3d(dtype=wp.int32),  # type: ignore[valid-type]
  rgb: wp.array4d(dtype=wp.uint8),  # type: ignore[valid-type]
  world_offset: int,
):
  """Unpack HyperLux's packed RGBA bytes directly into final RGB storage."""
  world, camera, pixel = wp.tid()  # type: ignore[attr-defined]
  value = wp.uint32(packed[world, camera, pixel])
  rgb[camera, world_offset + world, pixel, 0] = wp.uint8(
    value & wp.uint32(0xFF)  # type: ignore[operator]
  )
  rgb[camera, world_offset + world, pixel, 1] = wp.uint8(
    (value >> wp.uint32(8)) & wp.uint32(0xFF)  # type: ignore[operator]
  )
  rgb[camera, world_offset + world, pixel, 2] = wp.uint8(
    (value >> wp.uint32(16)) & wp.uint32(0xFF)  # type: ignore[operator]
  )


@wp.kernel
def _pack_cameras(
  ids: wp.array(dtype=int),  # type: ignore[valid-type]
  sensor_size: wp.array(dtype=wp.vec2),  # type: ignore[valid-type]
  xpos: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
  xmat: wp.array2d(dtype=wp.mat33),  # type: ignore[valid-type]
  fovy: wp.array2d(dtype=float),  # type: ignore[valid-type]
  intrinsic: wp.array2d(dtype=wp.vec4),  # type: ignore[valid-type]
  aspect: float,
  world_offset: int,
  cameras: wp.array3d(dtype=wp.vec3),  # type: ignore[valid-type]
):
  world, index = wp.tid()  # type: ignore[attr-defined]
  source = (world_offset + world) % xpos.shape[0]
  cam = ids[index]
  size = sensor_size[index]
  half_h = wp.tan(fovy[source, cam] * wp.pi / 360.0)
  left, top = -half_h * aspect, half_h
  span_x, span_y = 2.0 * half_h * aspect, 2.0 * half_h
  if size[1] != 0.0:
    width, height = size[0], size[1]
    if aspect > width / height:
      height = width / aspect
    else:
      width = height * aspect
    intr = intrinsic[source, cam]
    left = -(width * 0.5 - intr[2]) / intr[0]
    top = (height * 0.5 - intr[3]) / intr[1]
    span_x, span_y = width / intr[0], height / intr[1]
  rotation = xmat[source, cam]
  cameras[world, index, 0] = xpos[source, cam]
  cameras[world, index, 1] = rotation * wp.vec3(left, top, -1.0)
  cameras[world, index, 2] = rotation * wp.vec3(span_x, 0.0, 0.0)
  cameras[world, index, 3] = rotation * wp.vec3(0.0, -span_y, 0.0)


def _object(model: mujoco.MjModel, geom: int, textures: bool, fidelity: str):
  """Convert a geom in its local frame; textures travel through glTF once."""
  kind = model.geom_type[geom]
  mesh = (
    mujoco_mesh_to_trimesh(model, geom)
    if kind == mujoco.mjtGeom.mjGEOM_MESH
    else create_primitive_mesh(model, geom)
  )
  if kind == mujoco.mjtGeom.mjGEOM_PLANE:
    # mjviser's plane is a thin box with smoothed corner normals. Use a flat
    # surface so lighting and shadow-ray offsets follow the actual plane normal.
    x, y = np.where(model.geom_size[geom, :2] > 0, model.geom_size[geom, :2], 10.0)
    mesh = trimesh.Trimesh(
      vertices=[[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]],
      faces=[[0, 1, 2], [0, 2, 3]],
      process=False,
    )
  elif kind == mujoco.mjtGeom.mjGEOM_BOX:
    # A shared-vertex cube gets smoothed corner normals from trimesh. Boxes
    # need discontinuous face normals: smoothing makes room walls look curved
    # under point lights and biases the origins of shadow/reflection rays.
    vertices = np.asarray(mesh.vertices)[mesh.faces.ravel()]
    normals = np.repeat(mesh.face_normals, 3, axis=0)
    mesh = trimesh.Trimesh(
      vertices=vertices,
      faces=np.arange(len(vertices)).reshape(-1, 3),
      vertex_normals=normals,
      process=False,
    )
  mat = model.geom_matid[geom]
  rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[geom]
  metallic = float(model.mat_metallic[mat]) if mat >= 0 else 0.0
  roughness = float(model.mat_roughness[mat]) if mat >= 0 else 1.0
  # MuJoCo uses -1 for unspecified PBR fields; use HyperLux's material defaults.
  if metallic < 0:
    metallic = 0.0
  if roughness < 0:
    roughness = 1.0
  emission = float(model.mat_emission[mat]) if mat >= 0 else 0.0

  # MuJoCo planes use planar UVs and repeat factors. mjviser's primitive
  # conversion supplies geometry but no texture coordinates for planes.
  if textures and mat >= 0 and kind == mujoco.mjtGeom.mjGEOM_PLANE:
    tex = int(model.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB])
    if tex >= 0 and model.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D:
      h, w, c = model.tex_height[tex], model.tex_width[tex], model.tex_nchannel[tex]
      adr = model.tex_adr[tex]
      pixels = model.tex_data[adr : adr + h * w * c].reshape(h, w, c)
      uv = mesh.vertices[:, :2].copy()
      if not model.mat_texuniform[mat]:
        # Zero-sized MuJoCo planes become 20 m meshes in the conversion.
        half_size = np.where(
          model.geom_size[geom, :2] > 0, model.geom_size[geom, :2], 10.0
        )
        uv /= half_size
      uv = (uv + 1.0) * 0.5 * model.mat_texrepeat[mat]
      mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.PBRMaterial(
          baseColorTexture=Image.fromarray(np.flipud(pixels)),
          baseColorFactor=rgba,
        ),
      )

  if textures and isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    material = cast(trimesh.visual.material.PBRMaterial, mesh.visual.material)
    # HyperLux resolves packed metallic/roughness images by the glTF material name.
    material.name = f"material_{mat}"
    material.metallicFactor = metallic
    material.roughnessFactor = roughness
    material.emissiveFactor = rgba[:3] * emission
    for role, field in (
      (mujoco.mjtTextureRole.mjTEXROLE_NORMAL, "normalTexture"),
      (mujoco.mjtTextureRole.mjTEXROLE_ORM, "metallicRoughnessTexture"),
    ):
      tex = int(model.mat_texid[mat, role])
      if tex >= 0 and model.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D:
        h, w, c = model.tex_height[tex], model.tex_width[tex], model.tex_nchannel[tex]
        adr = model.tex_adr[tex]
        pixels = model.tex_data[adr : adr + h * w * c].reshape(h, w, c)
        texture_image = Image.fromarray(np.flipud(pixels))
        setattr(material, field, texture_image)
        if role == mujoco.mjtTextureRole.mjTEXROLE_ORM:
          material.occlusionTexture = texture_image
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
      # The texture conversion expands each face corner. Preserve the authored
      # corner normals instead of recalculating faceted normals on those vertices.
      mid = model.geom_dataid[geom]
      start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
      normal_ids = model.mesh_facenormal[start : start + count].ravel()
      mesh.vertex_normals = model.mesh_normal[model.mesh_normaladr[mid] + normal_ids]
    # HyperLux's GLB loader maps (x,y,z) -> (x,-z,y). Undo it at export.
    mesh.apply_transform(
      np.array(
        [
          [1, 0, 0, 0],
          [0, 0, 1, 0],
          [0, -1, 0, 0],
          [0, 0, 0, 1],
        ]
      )
    )
    with TemporaryDirectory(prefix="mjlab-hyperlux-") as directory:
      path = Path(directory) / "geom.glb"
      mesh.export(path)
      return hyperlux.load_object(path, fidelity=fidelity)

  obj = hyperlux.Object(model.geom(geom).name)
  obj.add_mesh(
    hyperlux.Mesh(
      np.asarray(mesh.vertices, dtype=np.float32),
      np.asarray(mesh.faces, dtype=np.int32),
      color=tuple(rgba[:3]),
      normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
      metallic=metallic,
      roughness=roughness,
      emissive=tuple(rgba[:3] * emission),
    )
  )
  return obj


def _scene(model: mujoco.MjModel, background: tuple[float, float, float]):
  """Import fixed lights. RGB materials and lighting are authored at startup."""
  scene = hyperlux.Scene()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  ambient = model.light_ambient[model.light_active != 0].sum(axis=0)
  if model.vis.headlight.active:
    ambient = ambient + model.vis.headlight.ambient
  scene.set_environment(ambient=tuple(ambient), background=background)
  for light in range(model.nlight):
    if not model.light_active[light]:
      continue
    color = tuple(model.light_diffuse[light])
    direction = tuple(data.light_xdir[light])
    if model.light_type[light] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
      # HyperLux expects a direction toward the light, MuJoCo away from it.
      source = hyperlux.SceneLight.directional(tuple(-data.light_xdir[light]), color)
    elif model.light_type[light] == mujoco.mjtLightType.mjLIGHT_SPOT:
      source = hyperlux.SceneLight.spot(
        tuple(data.light_xpos[light]),
        direction,
        color,
        cos_inner_cone=float(np.cos(np.deg2rad(model.light_cutoff[light]) * 0.8)),
        cos_outer_cone=float(np.cos(np.deg2rad(model.light_cutoff[light]))),
        attenuation=tuple(model.light_attenuation[light]),
      )
    else:
      source = hyperlux.SceneLight.point(
        tuple(data.light_xpos[light]),
        color,
        attenuation=tuple(model.light_attenuation[light]),
      )
    scene.add_light(source)
  return scene


def _to_cpu_array(value: object) -> np.ndarray:
  """Read an initialization-time tensor without assuming its device."""
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().numpy()
  return np.asarray(value)


def _local_transform(model: mujoco.MjModel, geom: int) -> np.ndarray:
  """Return a geom's fixed transform in its MuJoCo body frame."""
  rotation = np.empty(9, dtype=np.float64)
  mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
  transform = np.empty((3, 4), dtype=np.float32)
  transform[:, :3] = rotation.reshape(3, 3)
  transform[:, 3] = model.geom_pos[geom]
  return transform


def _static_transform(data: mjwarp.Data, geom: int, name: str) -> np.ndarray:
  """Return a shared geom transform, rejecting per-environment placement drift."""
  positions = _to_cpu_array(data.geom_xpos[:, geom])
  rotations = _to_cpu_array(data.geom_xmat[:, geom]).reshape(-1, 3, 3)
  if not np.allclose(positions, positions[0], rtol=0.0, atol=1e-6) or not np.allclose(
    rotations, rotations[0], rtol=0.0, atol=1e-6
  ):
    raise ValueError(
      f"HyperLux geom {name!r} is configured as static but has different "
      "initial poses across environments; mark it dynamic instead."
    )
  transform = np.empty((3, 4), dtype=np.float32)
  transform[:, :3] = rotations[0]
  transform[:, 3] = positions[0]
  return transform


def _split_geoms(
  mj_model: mujoco.MjModel,
  geom_ids: np.ndarray,
  static_patterns: tuple[str, ...],
  dynamic_patterns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
  """Apply user-owned static/dynamic labels to the selected MuJoCo geoms."""
  static_ids: list[int] = []
  dynamic_ids: list[int] = []
  for geom in geom_ids:
    name = mj_model.geom(int(geom)).name
    is_static = any(fnmatchcase(name, pattern) for pattern in static_patterns)
    is_dynamic = any(fnmatchcase(name, pattern) for pattern in dynamic_patterns)
    if is_static and is_dynamic:
      raise ValueError(
        f"HyperLux geom {name!r} matches both hyperlux_static_geoms and "
        "hyperlux_dynamic_geoms."
      )
    (static_ids if is_static else dynamic_ids).append(int(geom))
  return np.asarray(static_ids, dtype=int), np.asarray(dynamic_ids, dtype=int)


def _resolve_fidelity(cfg: HyperLuxCameraSensorCfg) -> str:
  return cfg.hyperlux_fidelity or ("veryhigh" if cfg.use_shadows else "high")


def _resolve_batch_size(cfg: HyperLuxCameraSensorCfg, device: Device) -> int:
  value = cfg.hyperlux_batch_size
  if isinstance(value, int):
    if type(value) is not int or value < 1:
      raise ValueError("hyperlux_batch_size must be a positive integer or 'auto'.")
    return value
  if value != "auto":
    raise ValueError("hyperlux_batch_size must be a positive integer or 'auto'.")

  # Batch 128 is the conservative profile for the project's RTX 4090 and
  # remains predictable for other GPUs, resolutions, and sampling settings.
  return 128


def _build_scene(
  mj_model: mujoco.MjModel,
  data: mjwarp.Data,
  cfg: HyperLuxCameraSensorCfg,
  static_geom_ids: np.ndarray,
  fidelity: str,
) -> hyperlux.Scene:
  scene = _scene(mj_model, cfg.hyperlux_background)
  for geom in static_geom_ids:
    name = mj_model.geom(int(geom)).name
    scene.add_object(
      _object(mj_model, int(geom), cfg.use_textures, fidelity),
      _static_transform(data, int(geom), name),
    )
  return scene


class HyperLuxRenderer:
  """One batch of cameras sharing resolution and sampling settings."""

  @classmethod
  def create_groups(
    cls,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    sensors: list[CameraSensor],
    device: Device,
  ) -> list[HyperLuxRenderer]:
    groups: dict[
      tuple[
        int,
        int,
        int,
        int | str,
        tuple[float, float, float],
        tuple[str, ...],
        tuple[str, ...],
        str,
        bool,
      ],
      list[CameraSensor],
    ] = defaultdict(list)
    for sensor in sensors:
      cfg = cast(HyperLuxCameraSensorCfg, sensor.cfg)
      groups[
        (
          cfg.width,
          cfg.height,
          cfg.hyperlux_anti_aliasing_grid,
          cfg.hyperlux_batch_size,
          cfg.hyperlux_background,
          tuple(cfg.hyperlux_static_geoms),
          tuple(cfg.hyperlux_dynamic_geoms),
          _resolve_fidelity(cfg),
          cfg.use_textures,
        )
      ].append(sensor)
    scenes: dict[tuple[object, ...], hyperlux.Scene] = {}
    renderers = []
    for group in groups.values():
      cfg = cast(HyperLuxCameraSensorCfg, group[0].cfg)
      geom_ids = np.flatnonzero(
        np.isin(mj_model.geom_group, cfg.enabled_geom_groups)
        & (mj_model.geom_rgba[:, 3] > 0)
      )
      static_geom_ids, dynamic_geom_ids = _split_geoms(
        mj_model,
        geom_ids,
        cfg.hyperlux_static_geoms,
        cfg.hyperlux_dynamic_geoms,
      )
      fidelity = _resolve_fidelity(cfg)
      scene_key = (
        cfg.hyperlux_background,
        cfg.use_textures,
        tuple(cfg.enabled_geom_groups),
        tuple(static_geom_ids.tolist()),
        fidelity,
      )
      if scene_key not in scenes:
        scenes[scene_key] = _build_scene(mj_model, data, cfg, static_geom_ids, fidelity)
      renderers.append(
        cls(
          mj_model,
          model,
          data,
          group,
          device,
          scene=scenes[scene_key],
          static_geom_ids=static_geom_ids,
          dynamic_geom_ids=dynamic_geom_ids,
          batch_size=_resolve_batch_size(cfg, device),
          fidelity=fidelity,
        )
      )
    return renderers

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    sensors: list[CameraSensor],
    device: Device,
    *,
    scene: hyperlux.Scene | None = None,
    static_geom_ids: np.ndarray | None = None,
    dynamic_geom_ids: np.ndarray | None = None,
    batch_size: int | None = None,
    fidelity: str | None = None,
  ):
    cfg = cast(HyperLuxCameraSensorCfg, sensors[0].cfg)
    self.device = device
    self.sensors = sensors
    self.width, self.height = cfg.width, cfg.height
    self.nworld = data.nworld
    self.batch_size = min(
      data.nworld,
      _resolve_batch_size(cfg, device) if batch_size is None else batch_size,
    )
    self.ncam = len(sensors)
    cam_ids = [s.camera_idx for s in sensors]
    if np.any(
      mj_model.cam_projection[cam_ids] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE
    ):
      raise ValueError("HyperLux RGB requires perspective cameras.")
    if static_geom_ids is None or dynamic_geom_ids is None:
      geom_ids = np.flatnonzero(
        np.isin(mj_model.geom_group, cfg.enabled_geom_groups)
        & (mj_model.geom_rgba[:, 3] > 0)
      )
      static_geom_ids, dynamic_geom_ids = _split_geoms(
        mj_model,
        geom_ids,
        cfg.hyperlux_static_geoms,
        cfg.hyperlux_dynamic_geoms,
      )
    if fidelity is None:
      fidelity = _resolve_fidelity(cfg)
    if scene is None:
      scene = _build_scene(mj_model, data, cfg, static_geom_ids, fidelity)

    pool = hyperlux.AssetPool() if len(dynamic_geom_ids) else None
    assets = []
    body_ids = []
    if pool is not None:
      grouped_geoms: dict[int, list[int]] = defaultdict(list)
      for geom in dynamic_geom_ids:
        grouped_geoms[int(mj_model.geom_bodyid[int(geom)])].append(int(geom))
      for body, body_geoms in grouped_geoms.items():
        assembly = hyperlux.ObjectAssembly()
        for geom in body_geoms:
          assembly.add_part(
            _object(mj_model, geom, cfg.use_textures, fidelity),
            _local_transform(mj_model, geom),
          )
        assets.append(pool.add_assembly(assembly))
        body_ids.append(body)
    self.renderer = hyperlux.HyperLux(
      width=cfg.width,
      height=cfg.height,
      batch_size=self.batch_size,
      cameras=tuple(CameraSpec(str(i), (RGB(),)) for i in cam_ids),
      fidelity=fidelity,
      anti_aliasing_grid=cfg.hyperlux_anti_aliasing_grid,
      num_overlays=self.batch_size if assets else 0,
      max_overlay_instances=len(dynamic_geom_ids),
      max_overlays_per_camera=1 if assets else 0,
    )
    if pool is None:
      self.renderer.init(scene)
    else:
      self.renderer.init(scene, asset_pool=pool)
    self.placements = None
    if assets:
      placements = []
      for world in range(self.batch_size):
        placements.extend(self.renderer.spawn(world, asset) for asset in assets)
        for camera in range(self.ncam):
          self.renderer.attach(world * self.ncam + camera, world)
      self.placements = self.renderer.placement_batch(placements)
      self.renderer.update()

    torch_device = str(device)
    self.poses = torch.empty((self.batch_size, len(assets), 3, 4), device=torch_device)
    self.cameras = torch.empty((self.batch_size, self.ncam, 4, 3), device=torch_device)
    self.pose_inputs = (
      [
        wp.array(body_ids, dtype=int, device=device),
        wp.from_torch(data.xpos[:], dtype=wp.vec3),
        wp.from_torch(data.xmat[:], dtype=wp.mat33),
      ]
      if assets
      else None
    )
    self.pose_output = wp.from_torch(self.poses) if assets else None
    self.camera_inputs = [
      wp.array(cam_ids, dtype=int, device=device),
      wp.array(mj_model.cam_sensorsize[cam_ids], dtype=wp.vec2, device=device),
      wp.from_torch(data.cam_xpos[:], dtype=wp.vec3),
      wp.from_torch(data.cam_xmat[:], dtype=wp.mat33),
      wp.from_torch(model.cam_fovy[:]),
      wp.from_torch(model.cam_intrinsic[:], dtype=wp.vec4),
      cfg.width / cfg.height,
    ]
    self.camera_output = wp.from_torch(self.cameras, dtype=wp.vec3)
    self.rgb_storage = torch.empty(
      (self.ncam, data.nworld, cfg.height * cfg.width, 3),
      dtype=torch.uint8,
      device=torch_device,
    )
    self.rgb = {
      i: self.rgb_storage[index].view(data.nworld, cfg.height, cfg.width, 3)
      for index, i in enumerate(cam_ids)
    }
    self.packed_output = self.renderer.outputs[RGB()].packed
    self.rgb_output = wp.from_torch(self.rgb_storage, dtype=wp.uint8)
    self._render_stream = None
    self._pending_rgb = False

  def _pack(self, world_offset: int) -> None:
    """Pack already-computed world poses and perspective rays on their device."""
    if self.pose_inputs is not None:
      wp.launch(
        _pack_poses,
        dim=self.poses.shape[:2],
        inputs=[*self.pose_inputs, world_offset, self.pose_output],
        device=self.device,
      )
    wp.launch(
      _pack_cameras,
      dim=self.cameras.shape[:2],
      inputs=[*self.camera_inputs, world_offset, self.camera_output],
      device=self.device,
    )

  def render(self) -> None:
    stream = (
      wp.stream_to_torch(wp.get_stream(self.device)) if self.device.is_cuda else None
    )
    self._render_stream = stream
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
      handle = stream.cuda_stream if stream is not None else 0
      for start in range(0, self.nworld, self.batch_size):
        self._pack(start)
        self.renderer.set_cameras(self.cameras.reshape(-1, 12), stream=handle)
        if self.placements is not None:
          self.renderer.set_pose(
            self.placements,
            hyperlux.Pose(self.poses.reshape(-1, 3, 4)),
            stream=handle,
          )
          self.renderer.update_launch()
        self.renderer.render_launch()
        count = min(self.batch_size, self.nworld - start)
        packed = from_dlpack(self.packed_output.blocks[0])
        packed = packed.view(torch.int32).reshape(
          self.batch_size, self.ncam, self.height * self.width
        )
        wp.launch(
          _unpack_rgb_packed,
          dim=(count, self.ncam, self.height * self.width),
          inputs=[wp.from_torch(packed, dtype=wp.int32), self.rgb_output, start],
          device=self.device,
        )
    self._pending_rgb = stream is not None

  def wait_for_rgb(self) -> None:
    """Order the caller's current stream after the asynchronous RGB unpack."""
    if self._pending_rgb and self._render_stream is not None:
      torch.cuda.current_stream(self.device.ordinal).wait_stream(self._render_stream)
      self._pending_rgb = False

  def close(self) -> None:
    self.wait_for_rgb()
    self.renderer.synchronize()
    del self.renderer
