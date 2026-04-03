#!/usr/bin/env python
# %%
import os

import itk

from physiomotion4d.segment_chest_vista_3d import SegmentChestVista3D


output_dir = "./results"
max_image = itk.imread(os.path.join(output_dir, "slice_fixed.mha"))

# %%
seg = SegmentChestVista3D()
result = seg.segment(max_image, contrast_enhanced_study=True)
labelmap_image = result["labelmap"]
lung_mask = result["lung"]
heart_mask = result["heart"]
major_vessels_mask = result["major_vessels"]
bone_mask = result["bone"]
soft_tissue_mask = result["soft_tissue"]
other_mask = result["other"]
contrast_mask = result["contrast"]
itk.imwrite(
    labelmap_image,
    os.path.join(output_dir, "slice_fixed.all_mask_vista3d.mha"),
    compression=True,
)
