# SimReady asset attribution

Source: [NVIDIA SimReady assets](https://huggingface.co/datasets/nvidia/simready-assets),
dataset revision `524747d9b5e1f21b5b18b56041788c01d745f1a1`.
The selected packages each include a Creative Commons Attribution 4.0
International license. A copy is included in [LICENSE-simready.txt](LICENSE-simready.txt).

Assets used:

- `desk_a01` — desk
- `chair_office_fabric_manager_a01` — office chair
- `cabinet_file_a01` — filing cabinet
- `computer_monitor_office_a01` — monitor
- `computer_keyboard_a01` — keyboard
- `computer_mouse_office_b01` — mouse and mousepad
- `dishware_mug_coffee_a01` — coffee mug
- `rack_h3m_s01` — warehouse rack
- `pallet_a01` — wooden pallet
- `crate_blue_a01` — plastic crate
- `rack_h3m_s02` - second warehouse rack
- `crate_green_a01` - green crate
- `drum_plastic_blue_a01` - plastic drum
- `vehicle_forklift_blue_a01` - forklift
- `barrier_fence_metal_a02` - barrier
- `sofa_soft_b01` - lounge seat
- `table_metal_a02` - workbench with backboard
- `bottle_laboratory_a01` - laboratory bottle

Adaptations: USD visual meshes and transforms were converted to OBJ/MJCF,
polygons triangulated, per-corner normals and UV transforms preserved, and
textures reduced to at most 1024 pixels per side. USD Preview Surface color,
normal and packed ORM inputs were mapped into MuJoCo materials and HyperLux.
Glass is shown opaque; MDL-specific features and USD articulations are not
reproduced. Furniture uses simple authored collision proxies.

The room shell, windows, floor pattern, pinboard, lighting and arrangement were
authored for this demo. The closed factory shell, fixtures and arrangement were
also authored for this demo. The G1 robot comes from the existing mjlab robot
assets. The repository includes rendered comparisons only. It does not redistribute the downloaded or converted scene assets.
