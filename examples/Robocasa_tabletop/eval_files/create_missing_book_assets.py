"""Create local placeholder RoboCasa sketchfab assets missing from downloaded assets."""

from __future__ import annotations

import argparse
from pathlib import Path


def _cube_obj(*, sx: float, sy: float, sz: float) -> str:
    x, y, z = sx / 2.0, sy / 2.0, sz / 2.0
    vertices = [
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    ]
    faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (5, 1, 4, 8),
    ]
    lines = ["# local placeholder book mesh"]
    lines += [f"v {vx:.6f} {vy:.6f} {vz:.6f}" for vx, vy, vz in vertices]
    lines += ["vn 0 0 1"]
    lines += ["f " + " ".join(str(idx) for idx in face) for face in faces]
    return "\n".join(lines) + "\n"


def _model_xml(asset_name: str, *, sx: float, sy: float, sz: float, color: str) -> str:
    mesh_name = f"{asset_name}_0"
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    return f"""<mujoco model="{asset_name}">
  <asset>
    <mesh name="{mesh_name}" file="meshes/{mesh_name}.obj"/>
    <texture name="{asset_name}_tex" type="2d" file="textures/base_color.png"/>
    <material name="{asset_name}_mat" texture="{asset_name}_tex" specular="0.2" shininess="0.1"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="{mesh_name}_visual" group="1" type="mesh" mesh="{mesh_name}" material="{asset_name}_mat" contype="0" conaffinity="0"/>
        <geom name="{mesh_name}_collision" group="0" type="box" size="{hx:.6f} {hy:.6f} {hz:.6f}" rgba="{color}"/>
      </body>
      <site name="bottom_site" pos="0 0 -{hz:.6f}" rgba="1 1 1 0.5" size="0.005"/>
      <site name="top_site" pos="0 0 {hz:.6f}" rgba="1 1 1 0.5" size="0.005"/>
      <site name="horizontal_radius_site" pos="{hx:.6f} {hy:.6f} 0" rgba="1 1 1 0.5" size="0.005"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_asset_variant(
    *,
    root: Path,
    category: str,
    asset_name: str,
    sx: float,
    sy: float,
    sz: float,
    texture_bgr: tuple[int, int, int],
    color: str,
) -> None:
    import cv2
    import numpy as np

    asset_dir = root / category / asset_name
    meshes = asset_dir / "meshes"
    textures = asset_dir / "textures"
    meshes.mkdir(parents=True, exist_ok=True)
    textures.mkdir(parents=True, exist_ok=True)
    (asset_dir / "model.xml").write_text(
        _model_xml(asset_name, sx=sx, sy=sy, sz=sz, color=color),
        encoding="utf-8",
    )
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    texture[:, :] = texture_bgr
    cv2.imwrite(str(textures / "base_color.png"), texture)
    mesh_text = _cube_obj(sx=sx, sy=sy, sz=sz)
    collision_text = _cube_obj(sx=sx, sy=sy, sz=sz)
    (meshes / f"{asset_name}_0.obj").write_text(mesh_text, encoding="utf-8")
    for group in range(2):
        for part in range(6):
            (meshes / f"{asset_name}_0_collision_{group}_{part}.obj").write_text(
                collision_text, encoding="utf-8"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_root", type=Path, required=True)
    args = parser.parse_args()

    root = args.assets_root / "objects" / "sketchfab"
    root.mkdir(parents=True, exist_ok=True)

    for book_idx in range(5):
        _write_asset_variant(
            root=root,
            category="book",
            asset_name=f"book_{book_idx}",
            sx=0.15,
            sy=0.21,
            sz=0.024,
            texture_bgr=(38, 74, 123),
            color="0.45 0.25 0.12 1",
        )
    for pod_idx in range(5):
        _write_asset_variant(
            root=root,
            category="coffee_pod",
            asset_name=f"coffee_pod_{pod_idx}",
            sx=0.045,
            sy=0.045,
            sz=0.03,
            texture_bgr=(180, 180, 40),
            color="0.2 0.55 0.75 1",
        )
    (root / "LOCAL_PLACEHOLDER_NOTICE.txt").write_text(
        "These book assets were generated locally because the downloaded "
        "PhysicalAI-DigitalCousin sketchfab package does not contain book/book_0..4 "
        "or coffee_pod/coffee_pod_0..4, while the GR1 code registry references them.\n",
        encoding="utf-8",
    )
    print(f"Created placeholder sketchfab assets under {root}")


if __name__ == "__main__":
    main()
