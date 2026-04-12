# Airways 3D model — source and attribution

The 3D airway model in this folder is a **third-party asset**. They are downloaded from Sketchfab and used here under their original license for the purposes of this prototype.

## Source

**Anatomy of the airways:**
https://sketchfab.com/3d-models/anatomy-of-the-airways-ad7d7e16b98f421db0cda79f265fcc8d

- Publisher: E-learning UMCG (University of Groningen)
- Created by: Anna Sieben
- Content reviewers: Cyril Luman, Dr. Walter Noordzij
- Created in Pixologic ZBrush, based on CT data, dissection-room specimens, and references in anatomy textbooks
- Mesh stats: 237.1k triangles, 119.9k vertices

## License

**CC BY-NC-SA** (Creative Commons Attribution-NonCommercial-ShareAlike)

- Attribution required
- Non-commercial use only
- Modified versions must be released under the same license

License details: https://creativecommons.org/licenses/by-nc-sa/4.0/

## Files in this folder

- `fbx_format (original)/` — original FBX download from Sketchfab (the format used as the starting point in the BronchoDrift pipeline)
- `GLB_format/`, `glTF_format/`, `USDZ_format/` — alternate format conversions provided by Sketchfab, retained for reference

## Pipeline use

The FBX from `fbx_format (original)/` was imported into Blender (`airways_filled.blend`), where the bronchial tree was isolated from the surrounding lung surfaces and filled to a closed, watertight solid. The watertight solid is required for clean voxel-based skeletonization in `bronchodrift.centerline`. The resulting mesh (`airways_filled.obj`, exported one level up in `data/raw/`) is the input used throughout the BronchoDrift scripts.

Modifications to the original asset are released under the same CC BY-NC-SA license.
