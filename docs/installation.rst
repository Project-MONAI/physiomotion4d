============
Installation
============

This guide covers the installation of PhysioMotion4D and its dependencies.

Prerequisites
=============

System Requirements
-------------------

* **Python**: 3.10, 3.11, or 3.12
* **GPU**: NVIDIA GPU with CUDA 13 (default) or CUDA 12 — CPU-only installation is not supported
* **RAM**: 16GB minimum (32GB+ recommended for large datasets)
* **Storage**: 10GB+ for package and model weights
* **Visualization**: NVIDIA Omniverse (optional, for USD visualization)

Software Dependencies
---------------------

PhysioMotion4D relies on several key packages:

* **Medical Imaging**: ITK, TubeTK, MONAI, nibabel, PyVista
* **AI/ML**: PyTorch, CuPy (CUDA 13 default; CUDA 12 via ``[cuda12]`` extra), transformers, MONAI
* **Registration**: icon-registration, unigradicon
* **Visualization**: USD-core, PyVista
* **Segmentation**: TotalSegmentator, VISTA-3D models

Installation Methods
====================

Method 1: Install from PyPI (Recommended)
------------------------------------------

The simplest way to install PhysioMotion4D is from PyPI.

CUDA 13 install (recommended):

.. code-block:: bash

   uv pip install "physiomotion4d[cuda13]"

CUDA 12 install:

.. code-block:: bash

   uv pip install "physiomotion4d[cuda12]"

For development with NVIDIA NIM cloud services:

.. code-block:: bash

   pip install physiomotion4d[nim]

Method 2: Install from Source
------------------------------

For development or to get the latest features:

**Step 1: Clone the repository**

.. code-block:: bash

   git clone https://github.com/Project-MONAI/physiomotion4d.git
   cd physiomotion4d

**Step 2: Create virtual environment**

.. tabs::

   .. tab:: Linux/macOS

      .. code-block:: bash

         python -m venv venv
         source venv/bin/activate

   .. tab:: Windows

      .. code-block:: bash

         python -m venv venv
         venv\Scripts\activate

**Step 3: Install uv package manager** (optional but recommended)

.. code-block:: bash

   pip install uv

**Step 4: Install PhysioMotion4D**

With uv (CUDA 13):

.. code-block:: bash

   uv pip install -e ".[cuda13]"

With uv (CUDA 12):

.. code-block:: bash

   uv pip install -e ".[cuda12]"

Optional Dependencies
=====================

Development Tools
-----------------

To install development dependencies (testing, linting, formatting):

.. code-block:: bash

   pip install physiomotion4d[dev]

This includes:

* **ruff** (fast linting and formatting)
* **mypy** (type checking)
* **pytest, pytest-cov** (testing)
* **pre-commit** (git hooks for automatic checks)

.. note::
   As of 2026, PhysioMotion4D uses Ruff as the primary linter and formatter,
   replacing the previous black, isort, flake8, and pylint tools for improved
   speed and simplicity.

Documentation Tools
-------------------

To build documentation locally:

.. code-block:: bash

   pip install physiomotion4d[docs]

Testing Dependencies
--------------------

To run tests:

.. code-block:: bash

   pip install physiomotion4d[test]

Verify Installation
===================

After installation, verify that PhysioMotion4D is correctly installed:

.. code-block:: python

   import physiomotion4d
   from physiomotion4d import ProcessHeartGatedCT
   
   print(f"PhysioMotion4D version: {physiomotion4d.__version__}")

Expected output:

.. code-block:: text

   PhysioMotion4D version: 2025.05.0

Command-Line Tools
==================

PhysioMotion4D provides command-line interfaces that should be available after installation:

.. code-block:: bash

   # Check CLI is available
   physiomotion4d --help
   physiomotion4d-heart-gated-ct --help

GPU Setup
=========

CUDA Installation
-----------------

PhysioMotion4D requires an NVIDIA GPU. Two CUDA versions are supported:

* **CUDA 13** — installed when you use the ``[cuda13]`` extra (recommended)
* **CUDA 12** — installed when you use the ``[cuda12]`` extra

CPU-only installation is not supported.

If CUDA is not yet installed, download the CUDA Toolkit from
`NVIDIA's website <https://developer.nvidia.com/cuda-downloads>`_, then verify:

.. code-block:: bash

   nvcc --version
   nvidia-smi

PyTorch with CUDA
-----------------

The default install pulls PyTorch built against CUDA 13. The ``[cuda12]`` extra
sources PyTorch, torchvision, and torchaudio from
``https://download.pytorch.org/whl/cu128``. To verify the active version:

.. code-block:: python

   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")

Troubleshooting
===============

Common Issues
-------------

**Issue: CUDA out of memory**

Solution: Reduce batch sizes or process smaller images. Most PhysioMotion4D functions work with limited GPU memory.

**Issue: Import errors for ITK or VTK**

Solution: These packages sometimes require system dependencies. On Ubuntu:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0

**Issue: TotalSegmentator download fails**

Solution: TotalSegmentator downloads models on first use. Ensure you have:

* Stable internet connection
* Sufficient disk space (~2GB for models)
* Write permissions in the cache directory

**Issue: USD files not rendering in Omniverse**

Solution:

1. Ensure NVIDIA Omniverse is installed
2. Check USD file integrity with ``usdview`` (included with usd-core)
3. Verify file paths are accessible to Omniverse

Getting Help
------------

If you encounter issues:

1. Check the :doc:`troubleshooting` guide
2. Search `GitHub Issues <https://github.com/Project-MONAI/physiomotion4d/issues>`_
3. Open a new issue with:

   * Python version
   * CUDA version
   * Error messages
   * Minimal code to reproduce

Next Steps
==========

* Continue to :doc:`quickstart` for your first PhysioMotion4D workflow
* Explore :doc:`examples` for common use cases
* Read :doc:`cli_scripts/overview` for detailed command-line workflows

