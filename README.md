# Mipmap to COLMAP Converter

A Python utility for converting **Mipmap Desktop** reconstruction projects into a **COLMAP-compatible dataset** ready for import into **Lichtfeld Studio** or other COLMAP-based workflows.

The script automatically extracts camera parameters, camera poses, sparse point clouds, and undistorted images from a completed Mipmap project, producing a standard COLMAP directory structure with minimal user interaction.

---

## Features

* Interactive command-line interface
* Supports multiple `part_XX` reconstructions in a single project
* Automatically selects the **latest completed task** for each part
* Supports multiple camera photogroups
* Converts camera intrinsics and poses to COLMAP format
* Extracts the reconstructed point cloud from `MipModel.ply`
* Optionally downsamples the point cloud for faster loading
* Copies all undistorted images into a flat `images/` directory
* Produces a complete COLMAP dataset ready for import into Lichtfeld Studio

---

## Output Structure

Each converted part is written to its own output directory:

```text
output_folder/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt
```

This structure is directly compatible with:

* Lichtfeld Studio
* COLMAP
* Gaussian Splatting pipelines
* Other software supporting COLMAP datasets

---

## Requirements

* Python 3.8+
* NumPy

Install the dependency:

```bash
pip install numpy
```

---

## Usage

Run the script:

```bash
python mipmap_to_colmap.py
```

The program will guide you through the conversion process.

### Step 1

Enter your Mipmap project folder.

Example:

```text
D:\mipmap-desktop\62fda2dd-9a1b-4e76-b369-6e7e0236ca0c
```

### Step 2

If multiple reconstruction parts are found, choose:

* Convert all parts
* Convert selected parts

### Step 3

Choose where to save the converted dataset.

You can either:

* create one base folder containing one output folder per part
* specify an individual output folder for each part

The script then converts every selected reconstruction automatically.

---

## What Gets Converted

For every reconstruction part the script exports:

### Cameras

* Camera model
* Resolution
* Focal lengths
* Principal point

Written as:

```
cameras.txt
```

---

### Images

For every photograph:

* Quaternion rotation
* Translation vector
* Camera ID
* Image filename

Written as:

```
images.txt
```

---

### Sparse Point Cloud

Reads:

```
MipModel.ply
```

Exports:

```
points3D.txt
```

If no point cloud exists, a minimal placeholder point cloud is generated so the dataset remains importable.

---

### Images

The script copies every undistorted image from

```text
result/.temp/undistort/
```

into

```text
images/
```

matching the filenames referenced by `images.txt`.

---

## Configuration

The only user-configurable parameter is the point cloud sampling ratio:

```python
PLY_KEEP_FRACTION = 0.2
```

Examples:

| Value | Result             |
| ----- | ------------------ |
| 1.0   | Keep all points    |
| 0.5   | Keep 50% of points |
| 0.2   | Keep 20% of points |
| 0.05  | Keep 5% of points  |

Reducing the point count can significantly decrease project size and improve loading performance.

---

## Workflow

The converter performs the following steps automatically:

1. Locate all reconstruction parts.
2. Find the latest completed reconstruction task.
3. Parse `mvs_undistort.xml`.
4. Extract camera intrinsics.
5. Extract camera poses.
6. Convert poses to COLMAP format.
7. Read `MipModel.ply`.
8. Optionally downsample the point cloud.
9. Write COLMAP sparse files.
10. Copy undistorted images.
11. Create a complete COLMAP dataset.

---

## Import into Lichtfeld Studio

For each exported dataset:

1. Open **Lichtfeld Studio**
2. Create a **New Project**
3. Select **Import Dataset**
4. Choose **COLMAP**
5. Select:

   * `sparse/0`
   * `images`

The project is ready for training or rendering.

---

## Current Limitations

* Supports the Mipmap Desktop project structure only.
* Uses the latest completed task for each reconstruction part.
* Assumes undistorted images are available in the default Mipmap output directory.
* Writes COLMAP text (`.txt`) files only (not binary `.bin` files).

---

## Future Improvements

Potential future additions include:

* Binary COLMAP export
* Command-line arguments (non-interactive mode)
* Progress bars
* Logging to file
* Automatic validation of project structure
* Optional coordinate system transforms
* Batch conversion of multiple projects

---

## License

MIT License

---

## Author

**Parsa Mahdavi**

If this project helps your work, consider giving the repository a ⭐.
