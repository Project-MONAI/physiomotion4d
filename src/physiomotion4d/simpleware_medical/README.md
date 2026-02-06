# Simpleware Medical Integration for PhysioMotion4D

This directory contains integration code for using Synopsys Simpleware Medical with PhysioMotion4D for heart segmentation.

## Overview

The integration enables PhysioMotion4D to leverage Simpleware Medical's ASCardio module for automated cardiac segmentation. The implementation uses a two-component architecture:

1. **segment_heart_simpleware.py** (in parent directory): A Python class that inherits from `SegmentChestBase` and manages the external Simpleware Medical process
2. **physiomotion_heart_segmentation.py** (this directory): A Python script that runs within the Simpleware Medical environment and performs the actual segmentation using ASCardio

## Requirements

- Synopsys Simpleware Medical X-2025.06 or later
- ASCardio module license
- Valid Simpleware Medical installation with console mode and Python scripting support
- ConsoleSimplewareMedical.exe (command-line version)

## Current Status

**✅ FUNCTIONAL**: This integration is ready for testing!

### Solution Implemented

The integration uses a pre-created `blank.sip` file to establish an active document in console mode:

1. **blank.sip** file is loaded at startup via `--input-file`
2. This provides an active document context for the Python script
3. Script can then import NIfTI images into the active document
4. ASCardio segmentation can proceed normally
5. Results are exported as NIfTI labelmaps

This approach enables full console mode automation without requiring GUI interaction.

### How It Works

```bash
ConsoleSimplewareMedical.exe \
    --input-file blank.sip \              # 🎯 Establishes active document
    --run-script script.py \               # Runs segmentation script
    --input-data params.txt \              # Passes parameters
    --exit-after-script                    # Exits when done
```

The Python script then:
```python
doc = sw.App.GetDocument()                 # ✅ Document is active
doc.ImportBackgroundFromNifti(input_path)  # ✅ Can import data
# ... perform ASCardio segmentation ...
# ... export results ...
```

## Installation

1. Install Simpleware Medical (default path: `C:\Program Files\Synopsys\Simpleware Medical\X-2025.06\`)
2. Ensure the ASCardio module is licensed and available
3. No additional Python packages are required (Simpleware has its own Python environment)

## Usage

### Basic Usage

```python
from physiomotion4d.segment_heart_simpleware import SegmentHeartSimpleware
import itk

# Create segmenter instance
segmenter = SegmentHeartSimpleware()

# Load CT image
ct_image = itk.imread("heart_ct.nii.gz")

# Perform segmentation
result = segmenter.segment(ct_image, contrast_enhanced_study=True)

# Access results
labelmap = result['labelmap']
heart_mask = result['heart']
vessel_mask = result['major_vessels']

# Save results
itk.imwrite(labelmap, "heart_segmentation.nii.gz")
```

### Custom Simpleware Path

If Simpleware Medical is installed in a non-default location:

```python
segmenter = SegmentHeartSimpleware()
segmenter.set_simpleware_executable_path(
    "D:/CustomPath/Simpleware/ConsoleSimplewareMedical.exe"
)
```

### Segmentation Output

The ASCardio module segments the following cardiac structures:

**Heart Structures (IDs 1-6):**
- 1: Left Ventricle
- 2: Right Ventricle
- 3: Left Atrium
- 4: Right Atrium
- 5: Myocardium
- 6: Left Atrial Appendage

**Major Vessels (IDs 10-14):**
- 10: Aorta
- 11: Pulmonary Artery
- 12: Superior Vena Cava
- 13: Inferior Vena Cava
- 14: Pulmonary Vein

## Architecture

### Process Flow

1. PhysioMotion4D preprocesses the CT image (resampling, intensity scaling)
2. Preprocessed image is saved to a temporary NIfTI file
3. ConsoleSimplewareMedical.exe is launched with the segmentation script:
   ```bash
   ConsoleSimplewareMedical.exe --run-script physiomotion_heart_segmentation.py \
       --input-data params.txt \
       --exit-after-script
   ```
   Where `params.txt` contains:
   ```
   input_file=input.nii.gz
   output_file=output.nii.gz
   ```
4. Script runs within Simpleware environment:
   - Loads the input image
   - Initializes ASCardio module
   - Runs automatic heart segmentation
   - Exports labelmap to NIfTI
5. PhysioMotion4D reads the output labelmap and returns results

### Communication

- **Executable**: `ConsoleSimplewareMedical.exe` (command-line version)
- **Script invocation**: `--run-script <script_path>`
- **Input parameters**: `--input-data <params_file>` (text file with key=value pairs)
- **Input**: NIfTI compressed image (`input_image.nii.gz`)
- **Output**: NIfTI compressed labelmap (`output_labelmap.nii.gz`)
- **Protocol**: File-based I/O via temporary directory
- **Timeout**: 10 minutes (configurable in code)

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Simpleware Medical executable not found`
- **Solution**:
  - Verify Simpleware installation path
  - Ensure you're using `ConsoleSimplewareMedical.exe` not `SimplewareMedical.exe`
  - Use `set_simpleware_executable_path()` to specify custom location

**Issue**: `unrecognised option '-python'`
- **Solution**: The GUI version `SimplewareMedical.exe` doesn't support command-line scripting. Use `ConsoleSimplewareMedical.exe` instead.

**Issue**: `ImportError: Failed to import Simpleware modules`
- **Solution**: Ensure the script is being called with `--run-script` flag through `ConsoleSimplewareMedical.exe`

**Issue**: `WARNING: No segmentation masks were created`
- **Solution**: Check input image quality, contrast, and field of view. Ensure the heart is visible in the scan.

**Issue**: Segmentation timeout after 600 seconds
- **Solution**: Image may be too large or high resolution. Consider adjusting preprocessing parameters.

### Logging

Enable detailed logging to troubleshoot issues:

```python
import logging

segmenter = SegmentHeartSimpleware(log_level=logging.DEBUG)
```

## Customization

### Modifying ASCardio Parameters

To customize ASCardio segmentation parameters, edit `physiomotion_heart_segmentation.py`:

```python
cardio.auto_segment(
    image=image,
    segment_chambers=True,
    segment_myocardium=True,
    segment_vessels=True,
    # Add custom parameters here
)
```

### Adding Custom Structures

To segment additional structures:

1. Update the `mask_id_mapping` dictionary in `physiomotion_heart_segmentation.py`
2. Update `heart_mask_ids` or `major_vessels_mask_ids` in `segment_heart_simpleware.py`

## Reference Documentation

For more information on Simpleware Medical and ASCardio:
- Simpleware Medical User Guide
- ASCardio Module Documentation
- Simpleware Python API Reference (ScriptingAPI.chm)
- Console Mode Documentation

Located in: `C:\Program Files\Synopsys\Simpleware Medical\X-2025.06\Documentation\`

### Console Mode Command Reference

```bash
# View all command-line options
ConsoleSimplewareMedical.exe --help

# Key options:
--run-script <script>       # Execute a Python script
--exit-after-script         # Close after script completes
--input-file <file>         # Open a Simpleware file
--input-data <file>         # Text file with data for script (key=value pairs)
--input-value <key=value>   # Single value for script (can only be used once)
--no-progress               # Disable progress messages
```

### Example Command

```bash
ConsoleSimplewareMedical.exe \
    --run-script physiomotion_heart_segmentation.py \
    --input-data params.txt \
    --exit-after-script \
    --no-progress
```

### Simpleware Python API Usage

Within the script, access command-line data using:

```python
import simpleware.scripting as sw

# Get App instance
app = sw.App.GetInstance()

# Read data from --input-data file
input_data_str = app.GetInput()

# Or read single value from --input-value
input_value_str = app.GetInputValue()
```

## License

This integration code is part of PhysioMotion4D and follows the same license.
Simpleware Medical and ASCardio are commercial products requiring separate licenses from Synopsys.
