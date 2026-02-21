"""
VTK to USD conversion workflow and batch runner.

Implements the pipeline from the Convert_VTK_To_USD experiment notebooks:
load one or more VTK files, optionally split by connectivity or cell type,
convert to USD, then apply a chosen appearance (solid color, anatomic material,
or colormap from a primvar with auto or specified intensity range).
"""

import logging
import re
from pathlib import Path
from typing import Literal


from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
from physiomotion4d.usd_anatomy_tools import USDAnatomyTools
from physiomotion4d.usd_tools import USDTools
from physiomotion4d.vtk_to_usd import (
    ConversionSettings,
    MaterialData,
    VTKToUSDConverter,
    read_vtk_file,
    validate_time_series_topology,
)


def discover_time_series(
    paths: list[Path],
    pattern: str = r"\.t(\d+)\.(vtk|vtp|vtu)$",
) -> list[tuple[int, Path]]:
    """Discover and sort time-series VTK files by extracted time index.

    Args:
        paths: List of paths to VTK files
        pattern: Regex with one group for time step number (default matches .t123.vtk)

    Returns:
        Sorted list of (time_step, path) tuples. If no match, returns [(0, p) for p in paths].
    """
    time_series: list[tuple[int, Path]] = []
    regex = re.compile(pattern, re.IGNORECASE)
    for p in paths:
        match = regex.search(p.name)
        if match:
            time_series.append((int(match.group(1)), Path(p)))
        else:
            time_series.append((0, Path(p)))
    time_series.sort(key=lambda x: (x[0], str(x[1])))
    return time_series


AppearanceKind = Literal["solid", "anatomy", "colormap"]


class WorkflowConvertVTKToUSD(PhysioMotion4DBase):
    """
    Workflow to convert one or more VTK files to USD with configurable
    splitting and appearance (solid color, anatomic material, or colormap).
    """

    def __init__(
        self,
        vtk_files: list[str | Path],
        output_usd: str | Path,
        *,
        separate_by_connectivity: bool = True,
        separate_by_cell_type: bool = False,
        mesh_name: str = "Mesh",
        times_per_second: float = 60.0,
        up_axis: str = "Y",
        triangulate: bool = True,
        extract_surface: bool = True,
        time_series_pattern: str = r"\.t(\d+)\.(vtk|vtp|vtu)$",
        appearance: AppearanceKind = "solid",
        solid_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
        anatomy_type: str = "heart",
        colormap_primvar: str | None = None,
        colormap_name: str = "viridis",
        colormap_intensity_range: tuple[float, float] | None = None,
        log_level: int | str = logging.INFO,
    ):
        """
        Initialize the VTK-to-USD workflow.

        Args:
            vtk_files: List of paths to VTK files (.vtk, .vtp, .vtu). One file = single frame;
                multiple files = time series (ordered by time_series_pattern).
            output_usd: Path to output USD file.
            separate_by_connectivity: If True, split mesh into separate objects by connectivity.
            separate_by_cell_type: If True, split mesh by cell type (triangle/quad/...).
                Cannot be True when separate_by_connectivity is True.
            mesh_name: Base name for the mesh (or first mesh when not splitting).
            times_per_second: FPS for time-varying data.
            up_axis: "Y" or "Z".
            triangulate: Triangulate meshes.
            extract_surface: For .vtu, extract surface before conversion.
            time_series_pattern: Regex to extract time index from filenames (one group).
            appearance: "solid" | "anatomy" | "colormap".
            solid_color: RGB in [0,1] when appearance == "solid".
            anatomy_type: Anatomy material name when appearance == "anatomy"
                (e.g. heart, lung, bone, soft_tissue).
            colormap_primvar: Primvar name for coloring when appearance == "colormap"
                (e.g. vtk_point_stress_c0). If None, a candidate is auto-picked when possible.
            colormap_name: Matplotlib colormap name when appearance == "colormap".
            colormap_intensity_range: Optional (vmin, vmax) for colormap; None = auto from data.
            log_level: Logging level.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.vtk_files = [Path(f) for f in vtk_files]
        self.output_usd = Path(output_usd)
        self.separate_by_connectivity = separate_by_connectivity
        self.separate_by_cell_type = separate_by_cell_type
        self.mesh_name = mesh_name
        self.times_per_second = times_per_second
        self.up_axis = up_axis
        self.triangulate = triangulate
        self.extract_surface = extract_surface
        self.time_series_pattern = time_series_pattern
        self.appearance = appearance
        self.solid_color = solid_color
        self.anatomy_type = anatomy_type
        self.colormap_primvar = colormap_primvar
        self.colormap_name = colormap_name
        self.colormap_intensity_range = colormap_intensity_range

        if separate_by_connectivity and separate_by_cell_type:
            raise ValueError(
                "separate_by_connectivity and separate_by_cell_type cannot both be True"
            )

    def run(self) -> str:
        """
        Run the full workflow: convert VTK to USD, then apply the chosen appearance.

        Returns:
            Path to the created USD file (str).
        """
        self.log_section("VTK to USD conversion workflow")

        if not self.vtk_files:
            raise ValueError("vtk_files must not be empty")

        # Discover time series
        time_series = discover_time_series(
            self.vtk_files, pattern=self.time_series_pattern
        )
        time_steps = [t for t, _ in time_series]
        paths_ordered = [p for _, p in time_series]
        n_frames = len(paths_ordered)

        self.log_info("Input: %d file(s), time steps: %s", n_frames, time_steps[:5])
        if n_frames > 5:
            self.log_info("  ... and %d more", n_frames - 5)
        self.log_info("Output: %s", self.output_usd)

        settings = ConversionSettings(
            triangulate_meshes=self.triangulate,
            compute_normals=False,
            preserve_point_arrays=True,
            preserve_cell_arrays=True,
            separate_objects_by_connectivity=self.separate_by_connectivity,
            separate_objects_by_cell_type=self.separate_by_cell_type,
            up_axis=self.up_axis,
            times_per_second=self.times_per_second,
            use_time_samples=True,
        )

        converter = VTKToUSDConverter(settings)
        default_material = MaterialData(
            name="default_material",
            diffuse_color=self.solid_color,
            use_vertex_colors=False,
        )

        if n_frames == 1:
            stage = converter.convert_file(
                paths_ordered[0],
                self.output_usd,
                mesh_name=self.mesh_name,
                material=default_material,
                extract_surface=self.extract_surface,
            )
        else:
            time_codes = [float(t) for t in time_steps]
            # Optional: validate topology consistency across frames
            try:
                mesh_sequence = [
                    read_vtk_file(p, extract_surface=self.extract_surface)
                    for p in paths_ordered
                ]
                report = validate_time_series_topology(mesh_sequence)
                if report.get("topology_changes"):
                    self.log_warning(
                        "Topology changes across %d frames",
                        len(report["topology_changes"]),
                    )
            except Exception as e:
                self.log_debug("Time series validation skipped: %s", e)

            stage = converter.convert_sequence(
                paths_ordered,
                self.output_usd,
                mesh_name=self.mesh_name,
                time_codes=time_codes,
                material=default_material,
                extract_surface=self.extract_surface,
            )

        # Post-process: apply chosen appearance to all meshes under /World/Meshes
        usd_tools = USDTools(log_level=self.log_level)
        mesh_paths = usd_tools.list_mesh_paths_under(
            str(self.output_usd), parent_path="/World/Meshes"
        )
        if not mesh_paths:
            self.log_warning("No mesh prims found under /World/Meshes")
            return str(self.output_usd)

        self.log_info(
            "Applying appearance '%s' to %d mesh(es)", self.appearance, len(mesh_paths)
        )

        if self.appearance == "solid":
            time_codes_for_color: list[float] | None = None
            if n_frames > 1 and stage.HasAuthoredTimeCodeRange():
                time_codes_for_color = [
                    float(t)
                    for t in range(
                        int(stage.GetStartTimeCode()),
                        int(stage.GetEndTimeCode()) + 1,
                    )
                ]
            for mesh_path in mesh_paths:
                usd_tools.set_solid_display_color(
                    str(self.output_usd),
                    mesh_path,
                    self.solid_color,
                    time_codes=time_codes_for_color,
                    bind_vertex_color_material=True,
                )

        elif self.appearance == "anatomy":
            anatomy_tools = USDAnatomyTools(stage, log_level=self.log_level)
            for mesh_path in mesh_paths:
                anatomy_tools.apply_anatomy_material_to_mesh(
                    mesh_path, self.anatomy_type
                )
            stage.Save()

        elif self.appearance == "colormap":
            primvar = self.colormap_primvar
            for mesh_path in mesh_paths:
                if primvar is None:
                    primvars = usd_tools.list_mesh_primvars(
                        str(self.output_usd), mesh_path
                    )
                    primvar = usd_tools.pick_color_primvar(primvars)
                if primvar is None:
                    self.log_warning(
                        "No color primvar found for %s; skip colormap", mesh_path
                    )
                    primvar = self.colormap_primvar
                    continue
                self.log_info(
                    "Applying colormap to %s from primvar %s", mesh_path, primvar
                )
                usd_tools.apply_colormap_from_primvar(
                    str(self.output_usd),
                    mesh_path,
                    primvar,
                    cmap=self.colormap_name,
                    intensity_range=self.colormap_intensity_range,
                    write_default_at_t0=True,
                    bind_vertex_color_material=True,
                )
                if self.colormap_primvar is None:
                    primvar = None  # next mesh: auto-pick again

        self.log_info("Workflow complete: %s", self.output_usd)
        return str(self.output_usd)
