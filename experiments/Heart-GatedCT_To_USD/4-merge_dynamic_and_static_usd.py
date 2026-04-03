#!/usr/bin/env python
# %%
import os

from physiomotion4d.usd_tools import USDTools

# %%
usd_tools = USDTools()

if os.path.exists("results/Slicer_CardiacGatedCT.merged_painted.usd"):
    os.remove("results/Slicer_CardiacGatedCT.merged_painted.usd")

usd_tools.merge_usd_files(
    "results/Slicer_CardiacGatedCT.merged_painted.usd",
    [
        "results/Slicer_CardiacGatedCT.dynamic_anatomy_painted.usd",
        "results/Slicer_CardiacGatedCT.static_anatomy_painted.usd",
    ],
)

if os.path.exists("results/Slicer_CardiacGatedCT.flattened_merged_painted.usd"):
    os.remove("results/Slicer_CardiacGatedCT.flattened_merged_painted.usd")

usd_tools.merge_usd_files_flattened(
    "results/Slicer_CardiacGatedCT.flattened_merged_painted.usd",
    [
        "results/Slicer_CardiacGatedCT.dynamic_anatomy_painted.usd",
        "results/Slicer_CardiacGatedCT.static_anatomy_painted.usd",
    ],
)
