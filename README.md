# Mipmap to COLMAP Converter

A Python utility that converts **Mipmap Desktop projects** into **COLMAP-compatible sparse reconstruction folders**, ready for use in tools like Lichtfeld Studio, COLMAP-based pipelines, or Gaussian Splatting workflows.

This script automates the conversion of complex multi-part Mipmap datasets into a structured COLMAP format, including camera intrinsics, poses, and reconstructed point clouds.

---

## Features

* Converts full Mipmap project directories into COLMAP sparse format
* Supports:

  * Multiple parts (`part_01`, `part_08`, etc.)
  * Multiple task versions (automatically selects latest)
  * Multiple photogroups / camera rigs
* Parses:

  * Camera intrinsics (focal length, principal point, aspect ratio)
  * Camera extrinsics (rotation + center → COLMAP pose format)
* Converts point cloud from `MipModel.ply`
* Optional point cloud downsampling for performance
* Applies coordinate system correction (−90° X-axis rotation) for compatibility with Lichtfeld / COLMAP conventions
* Outputs ready-to-use COLMAP structure

---

## Output Structure

After running the script, the following structure is generated inside the original project folder:

```
project_folder/
└── lichtfeld_export/
    └── part_XX/
        └── sparse/
            └── 0/
                ├── cameras.txt
                ├── images.txt
                └── points3D.txt
```

---

## Installation

you can copy the .py file in your mipmap project filder and run it with python and then copy the undistorted image folder from the mipmap project folder in the lichtfeld output folder and import it in the lichtfeld studio.

Clone the repository:

```bash
git clone https://github.com/your-username/mipmap-to-colmap.git
cd mipmap-to-colmap
```

Install dependencies:

```bash
pip install numpy
```

---

## Usage

Run the script by pointing it to a Mipmap project folder:

```bash
python mipmap_to_colmap.py <path_to_project_folder>
```

### Example

```bash
python mipmap_to_colmap.py D:\mipmap-desktop\62fda2dd-9a1b-4e76-b369-6e7e0236ca0c
```

---

## Configuration

You can adjust key reconstruction settings inside the script:

```python
PLY_KEEP_FRACTION = 0.2   # Keep fraction of point cloud (0–1)
APPLY_X_ROTATION = True   # Fix coordinate system alignment
```

---

## How It Works

### 1. Project Parsing

* Detects all parts and selects latest processing task
* Extracts photogroups and images from XML metadata

### 2. Camera Conversion

* Converts intrinsics into COLMAP PINHOLE format
* Converts rotation matrices → quaternions
* Computes translation vectors from camera centers

### 3. Coordinate System Alignment

* Applies −90° X-axis rotation to match COLMAP / Lichtfeld coordinate conventions

### 4. Point Cloud Conversion

* Reads binary or ASCII PLY (`MipModel.ply`)
* Extracts XYZ + RGB
* Optionally downsamples points

---

## Requirements

* Python 3.8+
* NumPy

---

## Notes

* Designed specifically for Mipmap Desktop export structure
* Tested with multi-camera photogroup datasets
* Optimized for downstream Gaussian Splatting and COLMAP workflows
* Large datasets may benefit from lowering `PLY_KEEP_FRACTION`

---

## Limitations

* Does not perform image undistortion (expects upstream correction if needed)
* Assumes consistent XML structure from Mipmap exports
* Basic PLY parsing (no advanced attribute support beyond RGB + XYZ)

---

## Author

Parsa Mahdavi

* add a **requirements.txt + setup.py**
* or refactor your script into a cleaner **modular package for thesis-level presentation**
