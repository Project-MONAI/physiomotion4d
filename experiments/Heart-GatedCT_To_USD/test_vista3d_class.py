#!/usr/bin/env python
# %%
import os

import itk

from physiomotion4d.segment_chest_vista_3d import SegmentChestVista3D

_HERE = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(_HERE, "results")
max_image = itk.imread(os.path.join(output_dir, "slice_fixed.mha"))

# %%
seg = SegmentChestVista3D()
result = seg.segment(max_image, contrast_enhanced_study=True)
labelmap_image = result["labelmap"]
itk.imwrite(
    labelmap_image,
    os.path.join(output_dir, "slice_fixed.all_mask_vista3d.mha"),
    compression=True,
)
