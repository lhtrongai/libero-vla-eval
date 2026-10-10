"""Make a copy of a LIBERO assets folder whose textures render as before MuJoCo 3.3.3.

MuJoCo 3.3.3 and later read the sRGB chunk of PNG textures and render tagged textures darker. Four LIBERO
PNGs carry the chunk, among them the floor of every scene, so LIBERO-Object (robot and objects on the floor)
looks clearly darker than the images LIBERO's training data was rendered with. Dropping the chunk restores
the old rendering exactly while keeping the same MuJoCo version and physics.

The copy is made with hard links (no extra disk space); only the four PNGs are rewritten as new files, so
the source folder is never modified. Use the copy with run_lerobot_libero_logged.py via
LIBERO_ASSETS_DIR=<dst>. For a LIBERO repository whose code loads assets from its own package folder
(the OpenVLA setup), hard-link-copy the whole repository first and pass its assets folder as both src and
dst, then put the copy first on PYTHONPATH.

Usage: python make_linear_texture_assets.py <src assets dir> <dst assets dir>
"""
import hashlib
import os
import shutil
import struct
import sys

TAGGED = (
    "textures/light-gray-floor-tile.png",
    "textures/tile_grigia_caldera_porcelain_floor.png",
    "stable_hope_objects/salad_dressing/texture_map.png",
    "stable_hope_objects/new_salad_dressing/texture_map.png",
)


def without_srgb_chunk(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    out, i = [data[:8]], 8
    while i < len(data):
        length, chunk_type = struct.unpack(">I4s", data[i:i + 8])
        end = i + 12 + length
        if chunk_type != b"sRGB":
            out.append(data[i:end])
        i = end
    return b"".join(out)


def main(src, dst):
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    if not os.path.isdir(os.path.join(src, "textures")):
        sys.exit(f"{src} does not look like a LIBERO assets folder (no textures/)")
    if not os.path.exists(dst):
        shutil.copytree(src, dst, copy_function=os.link)
    for rel in TAGGED:
        path = os.path.join(dst, rel)
        if not os.path.exists(path):
            print(f"missing (skipped): {rel}")
            continue
        with open(path, "rb") as f:
            data = f.read()
        fixed = without_srgb_chunk(data)
        if fixed != data:
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(fixed)
            os.replace(tmp, path)  # new file: a hard link back to src is broken, src stays untouched
        print(f"{rel}: sRGB chunk {'removed' if fixed != data else 'already absent'}, "
              f"sha1 {hashlib.sha1(fixed).hexdigest()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
