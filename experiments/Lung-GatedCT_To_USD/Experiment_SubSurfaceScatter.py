#!/usr/bin/env python
# %%
import os

from pxr import Sdf, Usd, UsdGeom, UsdShade

# %%
os.makedirs("results_SubsurfaceScatter", exist_ok=True)

# %%
# Initialize stage and define material/shader paths
stage = Usd.Stage.CreateNew(
    "results_SubsurfaceScatter/Experiment_SubSurfaceScatter.usda"
)

scope = UsdGeom.Scope.Define(stage, "/World")

sphere1 = UsdGeom.Sphere.Define(stage, "/World/Sphere")
sphere1.CreateRadiusAttr().Set(10)

mtl_path = Sdf.Path("/World/Looks/OmniSurface_Subsurface")
shader_path = mtl_path.AppendPath("Shader")
vista3d_running = False
# Create material and shader prims
mtl = UsdShade.Material.Define(stage, mtl_path)
shader = UsdShade.Shader.Define(stage, shader_path)

# Configure MDL source
shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
shader.SetSourceAsset("OmniSurface.mdl", "mdl")  # Use subsurface-capable MDL
shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")

# Enable and configure subsurface scattering
shader.CreateInput("enable_diffuse_transmission", Sdf.ValueTypeNames.Bool).Set(True)
shader.CreateInput("subsurface_weight", Sdf.ValueTypeNames.Float).Set(
    0.8
)  # Intensity (0-1)
shader.CreateInput("subsurface_scattering_color", Sdf.ValueTypeNames.Color3f).Set(
    (0.8, 0.2, 0.1)
)  # RGB
shader.CreateInput("subsurface_scale", Sdf.ValueTypeNames.Float).Set(
    1.5
)  # Scattering depth

# Connect shader outputs to material
mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
mtl.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
# Get target prim and bind material
binding_api = UsdShade.MaterialBindingAPI.Apply(sphere1.GetPrim())
binding_api.Bind(mtl)

stage.GetRootLayer().Save()
