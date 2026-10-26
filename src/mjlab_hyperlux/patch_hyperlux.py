"""Apply the bundled HyperLux 0.1.8 OptiX shadow fix explicitly at setup time.

Run: python -m mjlab_hyperlux.patch_hyperlux
The renderer never patches dependencies automatically.
"""

import subprocess
from importlib.metadata import version
from pathlib import Path

from hyperlux._jit._patches import apply_patch_scripts
from hyperlux._jit._workspace import dependencies_root, source_root


def main() -> None:
  if version("hyperlux") != "0.1.8":
    raise RuntimeError("This source patch targets HyperLux 0.1.8.")
  patch = Path(__file__).parent / "patches" / "hyperlux-0.1.8-shadows.patch"
  command = ["git", "apply", str(patch)]
  root = source_root()
  applied = subprocess.run(
    [*command, "--reverse", "--check"], cwd=root, capture_output=True
  )
  if applied.returncode != 0:
    subprocess.run([*command, "--check"], cwd=root, check=True)
    subprocess.run(command, cwd=root, check=True)
  # HyperLux applies its OWL patch during fresh configuration. Existing cached
  # dependencies need the same patch before CMake rebuilds them on next use.
  for owl in dependencies_root().glob("*/owl-src"):
    apply_patch_scripts({"OWL": owl})
  print("HyperLux shadow patch applied; native code rebuilds on next use.")


if __name__ == "__main__":
  main()
