"""Synthetic tests for model-to-patient workflow helpers."""

from __future__ import annotations

import inspect
from typing import Any

import itk
import numpy as np
import pyvista as pv

from physiomotion4d.segment_heart_simpleware import SegmentHeartSimpleware
from physiomotion4d.workflow_convert_image_to_vtk import WorkflowConvertImageToVTK
from physiomotion4d.workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)


def test_auto_generate_labelmap_accumulates_multilabel_models(
    monkeypatch: Any,
) -> None:
    """Multi-model labelmaps accumulate label IDs instead of overwriting prior labels."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    model = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )
    workflow.patient_image = image
    masks = [
        np.pad(np.ones((1, 1, 1), dtype=np.uint8), ((0, 2), (0, 2), (0, 2))),
        np.pad(np.ones((1, 1, 1), dtype=np.uint8), ((2, 0), (2, 0), (2, 0))),
    ]

    def fake_create_mask_from_mesh(*_args: object, **_kwargs: object) -> Any:
        mask = itk.image_from_array(masks.pop(0))
        mask.CopyInformation(workflow.patient_image)
        return mask

    monkeypatch.setattr(
        workflow.contour_tools,
        "create_mask_from_mesh",
        fake_create_mask_from_mesh,
    )

    output = workflow._auto_generate_labelmap([model, model], dilate_mm=0.0)
    output_arr = itk.GetArrayFromImage(output)

    assert output_arr[0, 0, 0] == 1
    assert output_arr[2, 2, 2] == 2
    assert sorted(np.unique(output_arr).tolist()) == [0, 1, 2]


def test_transform_model_applies_staged_transform() -> None:
    """Transform helper updates mesh points with image shape (Z, Y, X) = (3, 3, 3)."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    model = pv.PolyData(points)
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )

    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetIdentity()
    transform.SetTranslation((1.0, 2.0, 3.0))
    workflow.icp_forward_point_transform = transform
    workflow.pca_coefficients = None
    workflow.use_m2m_registration = False
    workflow.use_m2i_registration = False

    output = workflow.transform_model()

    assert output is not None
    np.testing.assert_allclose(output.points, points + np.array([1.0, 2.0, 3.0]))


def test_fit_workflow_default_segmentation_method_is_trimmed_branches() -> None:
    """Default segmentation_method must match the KCL-Heart-Model fit contract."""
    default = (
        inspect.signature(WorkflowFitStatisticalModelToPatient.__init__)
        .parameters["segmentation_method"]
        .default
    )
    assert default == "HeartSimplewareTrimmedBranches"


def test_fit_workflow_routes_default_to_image_to_vtk_with_trimmed_branches(
    monkeypatch: Any,
) -> None:
    """When patient_models is omitted, the workflow must invoke
    WorkflowConvertImageToVTK with segmentation_method='HeartSimplewareTrimmedBranches'."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    template = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    heart_mesh = pv.PolyData(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )

    captured: dict[str, Any] = {}

    class _FakeConvertImageToVTK:
        def __init__(self, **kwargs: Any) -> None:
            captured["init_kwargs"] = kwargs

        def run_workflow(self, **kwargs: Any) -> dict[str, Any]:
            captured["run_kwargs"] = kwargs
            return {"meshes": {"heart": heart_mesh}}

    monkeypatch.setattr(
        "physiomotion4d.workflow_fit_statistical_model_to_patient."
        "WorkflowConvertImageToVTK",
        _FakeConvertImageToVTK,
    )

    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=template,
        patient_image=image,
    )

    assert captured["init_kwargs"]["segmentation_method"] == (
        "HeartSimplewareTrimmedBranches"
    )
    assert captured["run_kwargs"]["anatomy_groups"] == ["heart"]
    assert captured["run_kwargs"]["contrast_enhanced_study"] is False
    assert workflow.patient_models == [heart_mesh]


def test_image_to_vtk_segmenter_dispatch_for_trimmed_branches() -> None:
    """WorkflowConvertImageToVTK('HeartSimplewareTrimmedBranches') must
    instantiate SegmentHeartSimpleware with branch trimming enabled, while
    'HeartSimpleware' must leave it disabled."""
    trimmed = WorkflowConvertImageToVTK(
        segmentation_method="HeartSimplewareTrimmedBranches"
    )._create_segmenter()
    assert isinstance(trimmed, SegmentHeartSimpleware)
    assert trimmed._trim_branches is True

    plain = WorkflowConvertImageToVTK(
        segmentation_method="HeartSimpleware"
    )._create_segmenter()
    assert isinstance(plain, SegmentHeartSimpleware)
    assert plain._trim_branches is False


def test_transform_model_preserves_unstructured_grid_topology() -> None:
    """Transform helper preserves cells with image shape (Z, Y, X) = (3, 3, 3)."""
    image = itk.image_from_array(np.zeros((3, 3, 3), dtype=np.float32))
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([4, 0, 1, 2, 3])
    celltypes = np.array([pv.CellType.TETRA])
    model = pv.UnstructuredGrid(cells, celltypes, points)
    model.cell_data["label"] = np.array([3], dtype=np.uint8)
    model.point_data["weights"] = np.arange(model.n_points, dtype=np.float64)
    workflow = WorkflowFitStatisticalModelToPatient(
        template_model=model,
        patient_models=[model],
        patient_image=image,
    )

    transform = itk.AffineTransform[itk.D, 3].New()
    transform.SetIdentity()
    transform.SetTranslation((1.0, 2.0, 3.0))
    workflow.icp_forward_point_transform = transform
    workflow.pca_coefficients = None
    workflow.use_m2m_registration = False
    workflow.use_m2i_registration = False

    output = workflow.transform_model()

    assert isinstance(output, pv.UnstructuredGrid)
    assert output.n_cells == model.n_cells
    np.testing.assert_array_equal(output.celltypes, model.celltypes)
    np.testing.assert_array_equal(output.cell_data["label"], model.cell_data["label"])
    np.testing.assert_array_equal(
        output.point_data["weights"], model.point_data["weights"]
    )
    np.testing.assert_allclose(output.points, points + np.array([1.0, 2.0, 3.0]))
