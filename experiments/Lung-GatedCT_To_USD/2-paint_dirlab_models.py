#!/usr/bin/env python
# %%
from data_dirlab_4d_ct import DataDirLab4DCT
from pxr import Usd

from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator
from physiomotion4d.usd_anatomy_tools import USDAnatomyTools


case_names = DataDirLab4DCT().case_names

case_names = [case_names[4]]

output_dir = "./results"

# %%
seg = SegmentChestTotalSegmentator()

for anatomy in ["all", "static_anatomy", "dynamic_anatomy"]:
    for case_name in case_names:
        stage = Usd.Stage.Open(f"{output_dir}/{case_name}_{anatomy}_lungGated.usd")
        painter = USDAnatomyTools(stage)
        painter.enhance_meshes(seg)
        stage.Export(f"{output_dir}/{case_name}_{anatomy}_lungGated_painted.usd")
