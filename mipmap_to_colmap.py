"""
mipmap_to_colmap.py
====================
Converts any mipmap-desktop project folder into a COLMAP dataset
ready for import into Lichtfeld Studio.

Features:
  - Interactive prompts for project folder and output folder
  - Handles multiple parts and multiple Photogroups automatically
  - Copies undistorted images to a flat images/ folder
  - No coordinate rotation applied (uses mipmap poses as-is)

Output structure:
  <output_folder>/
      images/
          3.jpg
          4.jpg
          7.jpg
          ...
      sparse/
          0/
              cameras.txt
              images.txt
              points3D.txt

Requirements:
  pip install numpy
"""

import os
import sys
import struct
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────

# Fraction of PLY points to keep (1.0 = all, 0.2 = 20%).
PLY_KEEP_FRACTION = 0.2


# ── Quaternion from rotation matrix ──────────────────────────────────────────

def rot_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


# ── PLY parser ────────────────────────────────────────────────────────────────

def parse_ply(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        num_vertices = 0
        properties   = []
        is_binary_le = False
        is_binary_be = False

        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif any(line.startswith(f"property {t} {n}")
                     for t in ("float", "float32")
                     for n in ("x", "y", "z", "nx", "ny", "nz")):
                properties.append((line.split()[-1], "f"))
            elif any(line.startswith(f"property {t} {n}")
                     for t in ("uchar", "uint8")
                     for n in ("red", "green", "blue", "alpha")):
                properties.append((line.split()[-1], "B"))
            elif "binary_little_endian" in line:
                is_binary_le = True
            elif "binary_big_endian" in line:
                is_binary_be = True

        prop_names = [p[0] for p in properties]
        prop_fmts  = [p[1] for p in properties]
        endian     = "<" if is_binary_le else ">" if is_binary_be else None

        xs, ys, zs, rs, gs, bs = [], [], [], [], [], []

        if endian is None:
            for _ in range(num_vertices):
                vals = f.readline().decode().split()
                row  = {prop_names[i]: float(vals[i]) for i in range(len(prop_names))}
                xs.append(row.get("x", 0))
                ys.append(row.get("y", 0))
                zs.append(row.get("z", 0))
                rs.append(int(row.get("red",   128)))
                gs.append(int(row.get("green", 128)))
                bs.append(int(row.get("blue",  128)))
        else:
            fmt         = endian + "".join(prop_fmts)
            record_size = struct.calcsize(fmt)
            raw         = f.read(num_vertices * record_size)
            for i in range(num_vertices):
                chunk = raw[i * record_size:(i + 1) * record_size]
                vals  = struct.unpack(fmt, chunk)
                row   = {prop_names[j]: vals[j] for j in range(len(prop_names))}
                xs.append(row.get("x", 0))
                ys.append(row.get("y", 0))
                zs.append(row.get("z", 0))
                rs.append(int(row.get("red",   128)))
                gs.append(int(row.get("green", 128)))
                bs.append(int(row.get("blue",  128)))

    return np.column_stack([xs, ys, zs]), np.column_stack([rs, gs, bs])


# ── XML parser ────────────────────────────────────────────────────────────────

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cameras = {}
    photos  = []

    for cam_id, pg in enumerate(root.findall(".//Photogroup"), start=1):
        w  = int(pg.findtext("ImageDimensions/Width"))
        h  = int(pg.findtext("ImageDimensions/Height"))
        fx = float(pg.findtext("FocalLengthPixels"))
        ar = float(pg.findtext("AspectRatio"))
        fy = fx * ar
        cx = float(pg.findtext("PrincipalPoint/x"))
        cy = float(pg.findtext("PrincipalPoint/y"))
        cameras[cam_id] = dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

        pg_photos = pg.findall("Photo")
        print(f"    Photogroup {cam_id}: {w}x{h}, fx={fx:.1f}, {len(pg_photos)} photos")

        for photo in pg_photos:
            img_id   = int(photo.findtext("Id"))
            img_name = os.path.basename(
                photo.findtext("ImagePath").replace("\\", "/")
            )

            pose = photo.find("Pose")
            if pose is None:
                continue

            rot = pose.find("Rotation")
            R = np.array([
                [float(rot.findtext("M_00")), float(rot.findtext("M_01")), float(rot.findtext("M_02"))],
                [float(rot.findtext("M_10")), float(rot.findtext("M_11")), float(rot.findtext("M_12"))],
                [float(rot.findtext("M_20")), float(rot.findtext("M_21")), float(rot.findtext("M_22"))],
            ])
            cen = pose.find("Center")
            C   = np.array([
                float(cen.findtext("x")),
                float(cen.findtext("y")),
                float(cen.findtext("z")),
            ])

            t = -R @ C
            qw, qx, qy, qz = rot_to_quat(R)

            photos.append({
                "id":     img_id,
                "cam_id": cam_id,
                "name":   img_name,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": t[0], "ty": t[1], "tz": t[2],
            })

    photos.sort(key=lambda p: p["id"])
    return cameras, photos


# ── COLMAP writers ────────────────────────────────────────────────────────────

def write_cameras(path, cameras):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id, c in cameras.items():
            f.write(
                f"{cam_id} PINHOLE {c['width']} {c['height']} "
                f"{c['fx']:.6f} {c['fy']:.6f} {c['cx']:.6f} {c['cy']:.6f}\n"
            )


def write_images(path, photos):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(photos)}\n")
        for p in photos:
            f.write(
                f"{p['id']} "
                f"{p['qw']:.9f} {p['qx']:.9f} {p['qy']:.9f} {p['qz']:.9f} "
                f"{p['tx']:.9f} {p['ty']:.9f} {p['tz']:.9f} "
                f"{p['cam_id']} {p['name']}\n\n"
            )


def write_points3d(path, ply_path):
    if ply_path and os.path.exists(ply_path):
        print(f"    Reading PLY: {os.path.basename(ply_path)}")
        xyz, rgb = parse_ply(ply_path)
        print(f"    Total PLY points: {len(xyz)}")

        if PLY_KEEP_FRACTION < 1.0:
            n   = max(1, int(len(xyz) * PLY_KEEP_FRACTION))
            idx = np.random.choice(len(xyz), n, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]
            print(f"    Downsampled to {len(xyz)} points ({PLY_KEEP_FRACTION*100:.0f}%)")

        with open(path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            f.write(f"# Number of points: {len(xyz)}\n")
            for i, (pt, cl) in enumerate(zip(xyz, rgb)):
                f.write(
                    f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{int(cl[0])} {int(cl[1])} {int(cl[2])} 0.0\n"
                )
    else:
        print(f"    No PLY found — writing dummy point")
        with open(path, "w") as f:
            f.write("# 3D point list\n# Number of points: 1\n")
            f.write("1 0.0 0.0 0.0 128 128 128 0.0\n")


# ── Image copy ────────────────────────────────────────────────────────────────

def copy_images(undist_dir, images_out_dir, expected_names):
    """Copy undistorted images to the flat images/ output folder."""
    os.makedirs(images_out_dir, exist_ok=True)
    copied  = 0
    missing = 0
    for name in expected_names:
        src = os.path.join(undist_dir, name)
        dst = os.path.join(images_out_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
    print(f"    Copied {copied} images to images/")
    if missing:
        print(f"    WARNING: {missing} image(s) not found in undistort folder")


# ── Project discovery ─────────────────────────────────────────────────────────

def find_latest_task_folder(part_dir):
    part_name  = os.path.basename(part_dir)
    candidates = []
    for entry in os.scandir(part_dir):
        if entry.is_dir() and entry.name.startswith(part_name + "-"):
            try:
                date_str = entry.name.split("-")[-1]
                date     = datetime.strptime(date_str, "%Y%m%d")
                candidates.append((date, entry.path))
            except ValueError:
                pass
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_parts(project_dir):
    parts = []
    for entry in os.scandir(project_dir):
        if entry.is_dir() and entry.name.startswith("part_"):
            task_folder = find_latest_task_folder(entry.path)
            if task_folder:
                parts.append((entry.name, task_folder))
    parts.sort(key=lambda x: x[0])
    return parts


# ── Part processor ────────────────────────────────────────────────────────────

def process_part(part_name, task_folder, output_dir):
    print(f"\n{'='*60}")
    print(f"  Part   : {part_name}")
    print(f"  Task   : {os.path.basename(task_folder)}")

    result_dir = os.path.join(task_folder, "result")
    xml_path   = os.path.join(result_dir, "AT", "mvs_undistort.xml")
    ply_path   = os.path.join(result_dir, "3D", "point-ply", "MipModel.ply")
    undist_dir = os.path.join(result_dir, ".temp", "undistort")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    images_dir = os.path.join(output_dir, "images")

    if not os.path.exists(xml_path):
        print(f"  SKIP — mvs_undistort.xml not found")
        return False

    if not os.path.exists(ply_path):
        print(f"  WARNING — MipModel.ply not found, will use dummy point cloud")
        ply_path = None

    if not os.path.exists(undist_dir):
        print(f"  WARNING — undistort folder not found: {undist_dir}")
        undist_dir = None

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Parse poses
    print(f"  Parsing XML...")
    cameras, photos = parse_xml(xml_path)
    print(f"  Cameras: {len(cameras)}, Photos with poses: {len(photos)}")

    if not photos:
        print(f"  SKIP — no photos with poses found")
        return False

    # Write COLMAP files
    write_cameras(os.path.join(sparse_dir, "cameras.txt"), cameras)
    print(f"  Written: cameras.txt ({len(cameras)} cameras)")

    write_images(os.path.join(sparse_dir, "images.txt"), photos)
    print(f"  Written: images.txt ({len(photos)} images)")

    print(f"  Writing points3D.txt...")
    write_points3d(os.path.join(sparse_dir, "points3D.txt"), ply_path)
    print(f"  Written: points3D.txt")

    # Copy images
    if undist_dir:
        print(f"  Copying images...")
        expected_names = [p["name"] for p in photos]
        copy_images(undist_dir, images_dir, expected_names)

    print(f"\n  Output folder : {output_dir}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def prompt(message, default=None):
    if default:
        result = input(f"{message} [{default}]: ").strip()
        return result if result else default
    else:
        while True:
            result = input(f"{message}: ").strip()
            if result:
                return result
            print("  This field is required.")


def main():
    print("=" * 60)
    print("  mipmap → COLMAP converter for Lichtfeld Studio")
    print("=" * 60)
    print()

    # Ask for project folder
    project_dir = prompt("Mipmap project folder").rstrip("\\/")
    if not os.path.isdir(project_dir):
        print(f"Error: folder not found: {project_dir}")
        sys.exit(1)

    # Find parts
    parts = find_parts(project_dir)
    if not parts:
        print("No part_XX folders with completed tasks found.")
        sys.exit(1)

    print(f"\nFound {len(parts)} part(s): {[p[0] for p in parts]}")

    # If multiple parts, ask which ones to process
    if len(parts) > 1:
        print("\nWhich parts do you want to convert?")
        print("  Enter part names separated by commas, or press Enter for ALL")
        print(f"  Available: {', '.join(p[0] for p in parts)}")
        selection = input("  > ").strip()
        if selection:
            selected = {s.strip() for s in selection.split(",")}
            parts    = [(n, t) for n, t in parts if n in selected]
            if not parts:
                print("No matching parts found.")
                sys.exit(1)
            print(f"Processing: {[p[0] for p in parts]}")

    # Ask for output folder (one per part or shared)
    print()
    if len(parts) == 1:
        default_out = os.path.join(project_dir, "lichtfeld_export", parts[0][0])
        output_dirs = {parts[0][0]: prompt("Output folder for Lichtfeld", default_out)}
    else:
        print("Output folder options:")
        print("  A) One folder per part  (you specify a base folder)")
        print("  B) Separate folder per part  (you specify each individually)")
        choice = input("  Choice [A/B]: ").strip().upper() or "A"
        output_dirs = {}
        if choice == "A":
            default_base = os.path.join(project_dir, "lichtfeld_export")
            base = prompt("Base output folder", default_base)
            for part_name, _ in parts:
                output_dirs[part_name] = os.path.join(base, part_name)
        else:
            for part_name, _ in parts:
                default_out = os.path.join(project_dir, "lichtfeld_export", part_name)
                output_dirs[part_name] = prompt(f"Output folder for {part_name}", default_out)

    # Process
    print()
    success = 0
    for part_name, task_folder in parts:
        out = output_dirs[part_name]
        if process_part(part_name, task_folder, out):
            success += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Done! Converted {success}/{len(parts)} part(s).\n")
    print("In Lichtfeld Studio for each part:")
    print("  New Project → Import Dataset → COLMAP")
    for part_name, _ in parts:
        out = output_dirs[part_name]
        print(f"\n  [{part_name}]")
        print(f"    Sparse : {os.path.join(out, 'sparse', '0')}")
        print(f"    Images : {os.path.join(out, 'images')}")


if __name__ == "__main__":
    main()
