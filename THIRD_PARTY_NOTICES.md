# Third-party notices

## HyperLux / RLtools

This project depends on [HyperLux](https://pypi.org/project/hyperlux/),
version 0.1.8. HyperLux is distributed under the MIT License. The project
also includes a small source patch that is applied to the installed HyperLux
OptiX sources to enable directional-light shadows. The patch is a derivative
change to that MIT-licensed code; the original HyperLux license and copyright
notice remain applicable to the dependency.

MIT License

Copyright (c) 2023 Jonas Eschmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This notice does not relicense HyperLux; it documents the license of the
separately installed dependency.

## mjlab

The RGB interface patch modifies [mjlab](https://github.com/mujocolab/mjlab),
which is licensed under Apache-2.0. The examples use mjlab's G1 robot through
the installed dependency; its mesh files are not included here.

## Comparison images

The rendered comparisons depict NVIDIA SimReady assets licensed under CC BY
4.0. Their source revision, asset names and adaptations are recorded in
[the media attribution](docs/media/ATTRIBUTION.md). No SimReady scene files,
meshes or textures are included.
