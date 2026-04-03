#!/usr/bin/env python
# %%
import os

import itk
import pyvista as pv

from physiomotion4d.contour_tools import ContourTools
from physiomotion4d import ConvertVTKToUSD
from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.usd_anatomy_tools import USDAnatomyTools

_HERE = os.path.dirname(os.path.abspath(__file__))

# %%
output_dir = os.path.join(_HERE, "results")

base_name = "slice_fixed"
# base_name = "slice_max.reg_dynamic_anatomy"

project_name = "Slicer_CardiacGatedCT"

do_transform_contours = True


# %%
def transform_contours(contours, transform_filenames, base_name, output_dir):
    con = ContourTools()
    for i, transform_filename in enumerate(transform_filenames):
        forward_transform = itk.transformread(transform_filename)[0]
        print(f"Applying transform {transform_filename} to {base_name}")

        new_contours = con.transform_contours(
            contours, forward_transform, with_deformation_magnitude=True
        )
        new_contours.save(
            os.path.join(
                output_dir, f"slice_{i:03d}.reg_{base_name}_inv.{base_name}_mask.vtp"
            )
        )


def convert_contours(base_name, output_dir, project_name, compute_normals=False):
    files = [
        f"{output_dir}/slice_{i:03d}.reg_{base_name}_inv.{base_name}_mask.vtp"
        for i in range(21)
    ]
    seg = SegmentChestTotalSegmentator()
    all_mask_ids = seg.all_mask_ids

    polydata = [pv.read(f) for f in files]

    # For cardiac gated CT data with 21 time points (0-20), each frame = 1 second
    # so we use times_per_second=1.0 instead of the default 24.0
    converter = ConvertVTKToUSD(
        project_name,
        polydata,
        all_mask_ids,
        compute_normals=compute_normals,
        times_per_second=1.0,
    )
    stage = converter.convert(
        os.path.join(output_dir, f"{project_name}.{base_name}.usd"),
    )

    painter = USDAnatomyTools(stage)
    painter.enhance_meshes(seg)
    if os.path.exists(
        os.path.join(output_dir, f"{project_name}.{base_name}_painted.usd")
    ):
        os.remove(os.path.join(output_dir, f"{project_name}.{base_name}_painted.usd"))
    stage.Export(os.path.join(output_dir, f"{project_name}.{base_name}_painted.usd"))


# %%
dynamic_transform_filenames = [
    os.path.join(output_dir, f"slice_{i:03d}.reg_dynamic_anatomy.forward.hdf")
    for i in range(21)
]
static_transform_filenames = [
    os.path.join(output_dir, f"slice_{i:03d}.reg_static_anatomy.forward.hdf")
    for i in range(21)
]
all_transform_filenames = [
    os.path.join(output_dir, f"slice_{i:03d}.reg_all.forward.hdf") for i in range(21)
]

dynamic_anatomy_contours = pv.read(
    os.path.join(output_dir, f"{base_name}.dynamic_anatomy_mask.vtp")
)
static_anatomy_contours = pv.read(
    os.path.join(output_dir, f"{base_name}.static_anatomy_mask.vtp")
)
all_contours = pv.read(os.path.join(output_dir, f"{base_name}.all_mask.vtp"))

# %%
transform_contours(all_contours, all_transform_filenames, "all", output_dir)
transform_contours(
    dynamic_anatomy_contours, dynamic_transform_filenames, "dynamic_anatomy", output_dir
)
transform_contours(
    static_anatomy_contours, static_transform_filenames, "static_anatomy", output_dir
)

# %%
convert_contours("all", output_dir, project_name, compute_normals=True)
convert_contours("dynamic_anatomy", output_dir, project_name, compute_normals=True)
convert_contours("static_anatomy", output_dir, project_name, compute_normals=True)
