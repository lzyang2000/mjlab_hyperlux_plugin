# Setup

Tested on Linux with an RTX 4090, driver 575.57.08, Python 3.13,
Torch 2.14.1+cu126, MuJoCo and MuJoCo Warp 3.16.0, Warp 1.18.0+cu12,
and HyperLux 0.1.8. You need Git, uv, a C++ compiler, CMake and a CUDA toolkit
with `nvcc`. Point `CUDA_HOME` and `CUDACXX` at your installed toolkit if
you have more than one CUDA version.

## Install

From a clone of this repository:

```bash
git clone https://github.com/mujocolab/mjlab.git mjlab
git -C mjlab checkout 8bb2bc9e187797689ec82c78c35019afd2fd0376
git -C mjlab apply --check ../patches/mjlab-rgb-backends.patch
git -C mjlab apply ../patches/mjlab-rgb-backends.patch
uv sync --project mjlab --extra cu126
uv pip install --python mjlab/.venv/bin/python -e '.[examples,dev]'
```

The patch adds mjlab's RGB renderer interface. Installing the plugin on
unpatched mjlab is insufficient. The pinned commit avoids applying the patch
to a different version of that interface.

For the tested pre-580 driver, install Warp's CUDA 12 wheel:

```bash
uv pip install --python mjlab/.venv/bin/python \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
```

Selecting mjlab's `cu126` extra chooses Torch's CUDA build; it does not
choose Warp's CUDA build. Use a Warp wheel compatible with your driver.

Configure the renderer and apply its shadow patch:

```bash
export HYPERLUX_BACKEND=OPTIX
export HYPERLUX_CACHE_DIR="$HOME/tmp/cache/hyperlux"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR" "$HYPERLUX_CACHE_DIR"
uv run --project mjlab --no-sync python -m mjlab_hyperlux.patch_hyperlux
```

This patches HyperLux 0.1.8's installed OptiX sources for directional shadows
and the reflection-plus-shadow stack. It is safe to run again. Reapply it
after reinstalling HyperLux. The renderer does not patch dependencies at runtime.

Use `uv run --project mjlab --no-sync` for the commands below so a lockfile
sync does not replace the locally installed plugin or Warp wheel.

## Run the examples

```bash
uv run --project mjlab --no-sync python examples/windowed_office.py
uv run --project mjlab --no-sync python examples/windowed_office.py --backend mujoco_warp
uv run --project mjlab --no-sync python examples/windowed_office.py --num-envs 4096
```

The first two commands save a PNG, 24-frame GIF and timing JSON under
`outputs/windowed-office/`. The batch command saves a PNG and timing JSON.
The room uses built-in geometry; the G1 robot comes from mjlab. This small
example differs from the furnished comparison scenes in the README.

`examples/render_rgb.py` also compares RGB rendering and physics-plus-rendering
on a simple robot-and-floor scene. Neither example reproduces the furnished
scene benchmark numbers without those original scenes.

## Compilation

The first render builds native code. Cached builds are reused when resolution,
render batch size, camera count, output types, AA, fidelity and dynamic-overlay
capacity match. Meshes, textures and poses are runtime data, so a different
scene does not automatically require a rebuild. A different robot can change
the required overlay capacity. Rendering 4096 copies shares the compiled module.

Scene loading and acceleration-structure construction still happen at startup,
even when the native module is cached.

## Development checks

```bash
uv run --project mjlab --no-sync ruff check src tests examples
uv run --project mjlab --no-sync ruff format --check src tests examples
uv run --project mjlab --no-sync pyright
uv run --project mjlab --no-sync pytest tests
uv build
```

Renderer tests require CUDA and the patched OptiX backend. Geometry-role and
lighting tests also run on CPU. Run timed GPU workloads without competing jobs.
