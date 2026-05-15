"""Convert a 4D image into a sequence of 3D images.

Reads a 4D medical image and splits it along the temporal axis into individual
3D ITK volumes, preserving origin, spacing, and direction in RAS world space.

Two reader paths are used:

* ``.nrrd`` files (including Slicer ``.seq.nrrd`` heart sequences whose
  per-voxel vector dimension exceeds ITK Python's wrapped Vector sizes) go
  through ``pynrrd``.
* Every other format goes through ``itk.imread`` and is expected to be a true
  4-dimensional image (e.g. NIfTI ``.nii.gz`` with ``dim[0] == 4``).
"""

import logging
from pathlib import Path
from typing import Any, Union

import itk
import nrrd
import numpy as np

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase


class ConvertImage4DTo3D(PhysioMotion4DBase):
    """Split a 4D ITK image (X, Y, Z, T) into a list of 3D ITK images."""

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the 4D-to-3D image converter.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.img_3d: list[Any] = []

    def load_image_4d(self, filename: str) -> None:
        """Load a 4D image and split it into a list of 3D ITK images.

        ``.nrrd`` files (including Slicer ``.seq.nrrd`` heart sequences) are
        read with ``pynrrd`` because their per-voxel vector dimension exceeds
        the component counts wrapped in ITK Python.  Every other format is
        read with ``itk.imread`` and must already be a 4-dimensional image.

        The result is a (T, Z, Y, X) ndarray plus 3D origin / spacing /
        direction; each temporal slice becomes a standalone 3D ITK image in
        ``self.img_3d``.

        Args:
            filename: Path to a 4D image file.
        """
        if filename.lower().endswith(".nrrd"):
            data, header = nrrd.read(filename)

            # pynrrd returns the data in (T, X, Y, Z) order for a 4D NRRD.
            # ITK numpy views use (T, Z, Y, X) — transpose the spatial axes.
            arr_4d = np.ascontiguousarray(np.asarray(data).transpose(0, 3, 2, 1))

            origin_3d = np.asarray(header["space origin"], dtype=float)
            spacing_3d = np.array(
                [abs(header["space directions"][x + 1][x]) for x in range(3)],
                dtype=float,
            )
            direction_3d = np.array(
                [header["measurement frame"][x] for x in range(3)], dtype=float
            )
            space = header.get("space", "")
            if "right" in space:
                direction_3d[0][0] *= -1
            if "anterior" in space:
                direction_3d[1][1] *= -1
            if "inferior" in space:
                direction_3d[2][2] *= -1
        else:
            img_4d = itk.imread(filename)
            arr_4d = itk.array_view_from_image(img_4d)
            if arr_4d.ndim != 4:
                raise ValueError(
                    f"Expected a 4D image, got array shape {arr_4d.shape}: {filename}"
                )
            origin_3d = np.asarray(img_4d.GetOrigin())[:3]
            spacing_3d = np.asarray(img_4d.GetSpacing())[:3]
            direction_3d = itk.array_from_matrix(img_4d.GetDirection())[:3, :3]

        direction_matrix = itk.matrix_from_array(direction_3d)
        self.img_3d = []
        for t in range(arr_4d.shape[0]):
            # Copy so each 3D image owns its buffer independently.
            arr_3d = np.ascontiguousarray(arr_4d[t])
            img3d = itk.image_from_array(arr_3d)
            img3d.SetOrigin(origin_3d.tolist())
            img3d.SetSpacing(spacing_3d.tolist())
            img3d.SetDirection(direction_matrix)
            self.img_3d.append(img3d)

    def get_3d_image(self, index: int) -> Any:
        """Return the 3D ITK image at the given time index."""
        return self.img_3d[index]

    def get_number_of_3d_images(self) -> int:
        """Return the number of 3D images currently held."""
        return len(self.img_3d)

    def save_3d_images(
        self,
        directory: Union[str, Path],
        basename: str,
        suffix: str = "mha",
    ) -> None:
        """Write each held 3D image to ``{directory}/{basename}_{i:03d}.{suffix}``.

        Args:
            directory: Output directory; created if it does not exist.
            basename: Filename stem used for every saved volume.
            suffix: File extension (default: ``mha``).
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(self.get_number_of_3d_images()):
            itk.imwrite(
                self.img_3d[i],
                str(dir_path / f"{basename}_{i:03d}.{suffix}"),
                compression=True,
            )
