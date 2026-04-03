#!/usr/bin/env python
# %%
import os
import urllib.request

from physiomotion4d.convert_nrrd_4d_to_3d import ConvertNRRD4DTo3D

_HERE = os.path.dirname(os.path.abspath(__file__))

# %%
data_dir = os.path.join(_HERE, "..", "..", "data", "Slicer-Heart-CT")
output_dir = os.path.join(_HERE, "results")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
input_image_url = "https://github.com/SlicerHeart/SlicerHeart/releases/download/TestingData/TruncalValve_4DCT.seq.nrrd"
input_image_filename = os.path.join(data_dir, "TruncalValve_4DCT.seq.nrrd")

if not os.path.exists(input_image_filename):
    urllib.request.urlretrieve(input_image_url, input_image_filename)

# %%
if not os.path.exists(f"{data_dir}/slice_000.mha"):
    conv = ConvertNRRD4DTo3D()
    conv.load_nrrd_4d(f"{data_dir}/TruncalValve_4DCT.seq.nrrd")
    conv.save_3d_images(f"{data_dir}/slice")
