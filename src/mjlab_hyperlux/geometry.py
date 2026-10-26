"""MuJoCo box geometry and texture coordinates for mesh-based renderers."""

import mujoco
import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


def box_mesh(
  model: mujoco.MjModel, geom: int, textures: bool = True
) -> trimesh.Trimesh:
  """Build a box with sharp normals and MuJoCo Warp's per-face UV mapping."""
  mesh = trimesh.creation.box(extents=2 * model.geom_size[geom])
  vertices = np.asarray(mesh.vertices)[mesh.faces.ravel()]
  normals = np.repeat(mesh.face_normals, 3, axis=0)
  mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=np.arange(len(vertices)).reshape(-1, 3),
    vertex_normals=normals,
    process=False,
  )
  mat = int(model.geom_matid[geom])
  if not textures or mat < 0:
    return mesh
  tex = int(model.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB])
  if tex < 0 or model.tex_type[tex] != mujoco.mjtTexture.mjTEXTURE_2D:
    return mesh

  # Each face needs its own corners: adjacent faces have different UVs.
  x, y, z = (vertices / model.geom_size[geom]).T
  uv = np.empty((len(vertices), 2))
  for axis, sign, u, v in (
    (2, 1, x, -y),
    (2, -1, x, y),
    (0, 1, y, -z),
    (0, -1, -y, -z),
    (1, -1, x, -z),
    (1, 1, -x, -z),
  ):
    face = normals[:, axis] == sign
    uv[face] = np.column_stack((u[face], v[face])) * 0.5 + 0.5
  uv *= model.mat_texrepeat[mat]
  if model.mat_texuniform[mat]:
    uv *= model.geom_size[geom, :2]
  h, w, c = model.tex_height[tex], model.tex_width[tex], model.tex_nchannel[tex]
  start = model.tex_adr[tex]
  pixels = model.tex_data[start : start + h * w * c].reshape(h, w, c)
  mesh.visual = TextureVisuals(
    uv=uv,
    material=PBRMaterial(
      baseColorTexture=Image.fromarray(np.flipud(pixels)),
      baseColorFactor=model.mat_rgba[mat],
    ),
  )
  return mesh
