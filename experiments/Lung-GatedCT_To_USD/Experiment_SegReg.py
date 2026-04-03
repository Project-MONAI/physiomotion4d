#!/usr/bin/env python
# %%
import os

import itk

from data_dirlab_4d_ct import DataDirLab4DCT

from physiomotion4d import RegisterImagesICON
from physiomotion4d import SegmentChestTotalSegmentator
from physiomotion4d import SegmentChestVista3D

# %%
fixed_image = DataDirLab4DCT().fix_image(
    itk.imread("../../data/DirLab-4DCT/Case1Pack_T30.mhd")
)
moving_image = DataDirLab4DCT().fix_image(
    itk.imread("../../data/DirLab-4DCT/Case1Pack_T00.mhd")
)

# %%
# Register images
reg_images = RegisterImagesICON()
reg_images.set_fixed_image(fixed_image)
_ = reg_images.register(moving_image)
moving_image_registered = reg_images.get_registered_image()
os.makedirs("results_SegReg", exist_ok=True)
itk.imwrite(
    moving_image_registered, "results_SegReg/Experiment_reg.mha", compression=True
)

# %%
img = itk.imread("results_SegReg/Experiment_reg.mha")
tot_seg = SegmentChestTotalSegmentator()
seg_results = tot_seg.segment(img, contrast_enhanced_study=False)
itk.imwrite(
    seg_results["labelmap"], "results_SegReg/Experiment_totseg.mha", compression=True
)

# %%
# This section requires the Vista3D container to be running

vista3d_running = False
if vista3d_running:
    img = itk.imread("Experiment_SegReg/Experiment_reg.mha")

    tot_seg = SegmentChestVista3D()

    seg_image = tot_seg.segment(img, contrast_enhanced_study=False)

    itk.imwrite(
        seg_image[0], "Experiment_SegReg/Experiment_vista3d.mha", compression=True
    )
    itk.imwrite(
        seg_image[1], "Experiment_SegReg/Experiment_vista3d_lung.mha", compression=True
    )
    itk.imwrite(
        seg_image[2], "Experiment_SegReg/Experiment_vista3d_heart.mha", compression=True
    )
    itk.imwrite(
        seg_image[3], "Experiment_SegReg/Experiment_vista3d_bone.mha", compression=True
    )
    itk.imwrite(
        seg_image[4],
        "Experiment_SegReg/Experiment_vista3d_soft_tissue.mha",
        compression=True,
    )
    itk.imwrite(
        seg_image[5], "Experiment_SegReg/Experiment_vista3d_other.mha", compression=True
    )
    itk.imwrite(
        seg_image[6],
        "Experiment_SegReg/Experiment_vista3d_contrast.mha",
        compression=True,
    )
