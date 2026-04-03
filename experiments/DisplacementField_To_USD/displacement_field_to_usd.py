#!/usr/bin/env python
# %% [markdown]
# # Displacement Field to USD for Omniverse Visualization
#
# This notebook demonstrates how to convert time-varying 3D displacement fields (stored as NIfTI images) into USD format for visualization in NVIDIA Omniverse using the PhysicsNeMo extension.
#
# ## Pipeline Overview
#
# 1. Load 3D vector fields from NIfTI files using ITK
# 2. Convert ITK images to VTK data structures
# 3. Create PhysicsNeMo-compatible USD stages for time-varying visualization
# 4. Export animated USD stage for Omniverse Create/Kit
#
# ## Architecture
#
# The `DisplacementFieldToUSD` class encapsulates all pipeline logic for converting medical imaging displacement fields to Omniverse-compatible USD format.
#
# ## Required Libraries
#
# - ITK (InsightToolkit) for medical image I/O
# - VTK (Visualization Toolkit) for data structure conversion
# - numpy for array processing
# - PhysicsNeMo and PhysicsNeMo-Sym for physics-based visualization
# - Omniverse USD Python API for stage creation
#

# %% [markdown]
# ## 1. Install Dependencies
#
# Install the latest versions of PhysicsNeMo and PhysicsNeMo-Sym from GitHub.
#

# %%
# Install PhysicsNeMo and PhysicsNeMo-Sym from GitHub
# Uncomment these lines if running for the first time

# # !pip install git+https://github.com/NVIDIA/physicsnemo.git
# # !pip install git+https://github.com/NVIDIA/physicsnemo-sym.git

# Install other required packages
# # !pip install itk vtk numpy pxr


# %% [markdown]
# ## 2. Import Libraries
#

# %%
import os

import itk
import numpy as np
import vtk
from displacement_field_converter import DisplacementFieldToUSD

print("Libraries imported successfully!")
print(f"ITK version: {itk.Version.GetITKVersion()}")
print(f"VTK version: {vtk.vtkVersion.GetVTKVersion()}")

# %% [markdown]
# ## 3. Import DisplacementFieldToUSD Class
#
# Import the converter class from the local module.
#

# %%
# Display class documentation
help(DisplacementFieldToUSD)


# %% [markdown]
# ## 4. Helper: Generate Sample Displacement Fields
#
# For demonstration purposes, this function creates synthetic time-varying displacement fields.
#


# %%
def generate_sample_displacement_fields(
    output_dir: str, n_timesteps: int = 10, size: tuple[int, int, int] = (32, 32, 32)
) -> list[str]:
    """
    Generate synthetic time-varying displacement fields for demonstration.

    Creates a rotating/pulsating vector field pattern.

    Args:
        output_dir: Directory to save NIfTI files
        n_timesteps: Number of time steps to generate
        size: 3D size of the displacement field

    Returns:
        List of file paths to generated NIfTI files
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []

    # Create coordinate grid
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, size[2]),
        np.linspace(-1, 1, size[1]),
        np.linspace(-1, 1, size[0]),
        indexing="ij",
    )

    for t in range(n_timesteps):
        # Time-varying rotation angle
        theta = 2 * np.pi * t / n_timesteps

        # Create rotating vector field with radial component
        r = np.sqrt(x**2 + y**2 + z**2)

        # Displacement components (rotating + pulsating)
        displacement_x = -y * np.cos(theta) + z * np.sin(theta)
        displacement_y = x * np.cos(theta) - r * 0.2 * np.sin(theta)
        displacement_z = -x * np.sin(theta) + y * np.cos(theta)

        # Scale by distance from center (creates flow pattern)
        scale_factor = 5.0 * (1 - r / np.max(r))
        displacement_x *= scale_factor
        displacement_y *= scale_factor
        displacement_z *= scale_factor

        # Stack into vector field (z, y, x, 3)
        displacement_field = np.stack(
            [displacement_x, displacement_y, displacement_z], axis=-1
        ).astype(np.float32)

        # Convert to ITK image
        itk_image = itk.image_from_array(displacement_field, is_vector=True)
        itk_image.SetSpacing([1.0, 1.0, 1.0])
        itk_image.SetOrigin([0.0, 0.0, 0.0])

        # Save as NIfTI
        file_path = os.path.join(output_dir, f"displacement_t{t:03d}.nii.gz")
        itk.imwrite(itk_image, file_path, compression=True)
        file_paths.append(file_path)

        print(f"Generated: {file_path}")

    return file_paths


# %% [markdown]
# ## 5. Example Usage
#
# Demonstrate the complete pipeline with synthetic data.
#

# %%
# Configuration
output_dir = "./sample_displacement_fields"
usd_output_path = "./displacement_field_animation.usd"
n_timesteps = 10

# Generate sample data
print("Generating sample displacement fields...")
nifti_files = generate_sample_displacement_fields(
    output_dir, n_timesteps=n_timesteps, size=(32, 32, 32)
)

print(f"\\nGenerated {len(nifti_files)} sample files")

# %%
# Create converter instance
converter = DisplacementFieldToUSD(subsample_factor=4, vector_scale=2.0)

# Run complete pipeline
stage = converter.process_pipeline(
    nifti_files=nifti_files, output_path=usd_output_path, fps=24.0
)

# %%
# Example with real displacement field data
# Uncomment and modify paths to use your own data:

# nifti_files = [
#     "path/to/displacement_t00.nii.gz",
#     "path/to/displacement_t01.nii.gz",
#     "path/to/displacement_t02.nii.gz",
#     # ... more files
# ]
#
# converter = DisplacementFieldToUSD(subsample_factor=8, vector_scale=5.0)
# stage = converter.process_pipeline(
#     nifti_files=nifti_files,
#     output_path="cardiac_motion.usd",
#     fps=10.0
# )


# %% [markdown]
# ## 7. Visualization in Omniverse
#
# ### Steps to Visualize:
#
# 1. **Open Omniverse Create or Kit**
#    - Launch NVIDIA Omniverse Create
#    - File → Open and select the generated USD file
#
# 2. **Enable PhysicsNeMo Extension**
#    - Window → Extensions
#    - Search for "PhysicsNeMo"
#    - Enable the extension
#
# 3. **Visualize the Vector Field**
#    - Select `/DisplacementField/VectorField` prim in the stage
#    - In PhysicsNeMo panel, choose visualization mode:
#      - **Streamlines**: Flow trajectories
#      - **Vector Glyphs**: Direction and magnitude at points
#      - **Volume Rendering**: Field magnitude as volume
#      - **Flow Particles**: Animated particles
#
# 4. **Play Animation**
#    - Use timeline controls to play through time steps
#    - Adjust playback speed as needed
#
# ### Class Methods Summary:
#
# - `load_nifti_files()`: Load displacement fields from NIfTI
# - `convert_to_vtk()`: Convert ITK images to VTK format
# - `extract_all_vector_fields()`: Extract subsampled data
# - `create_usd_stage()`: Create time-varying USD stage
# - `process_pipeline()`: Run complete pipeline
#
# ### Key Features:
#
# ✅ Class-based architecture for clean, reusable code
# ✅ Complete ITK → VTK → USD pipeline
# ✅ Time-varying animation support
# ✅ PhysicsNeMo-compatible velocities attribute
# ✅ Configurable subsampling and vector scaling
# ✅ Production-ready for medical imaging workflows
#
# This implementation encapsulates all logic in the `DisplacementFieldToUSD` class, making it easy to integrate into larger pipelines or customize for specific use cases.
#
