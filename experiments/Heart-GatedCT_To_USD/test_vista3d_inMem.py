#!/usr/bin/env python
# %%
import itk
import numpy as np
import torch


def vista3d_inference_from_itk(
    itk_image,
    label_prompt=None,
    points=None,
    point_labels=None,
    device=None,
    bundle_path=None,
    model_cache_dir=None,
):
    # 1. Import dependencies
    import itk
    from monai.bundle import download
    from monai.data.itk_torch_bridge import itk_image_to_metatensor
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import vista3d132
    from monai.transforms import (
        CropForeground,
        EnsureChannelFirst,
        EnsureType,
        ScaleIntensityRange,
        Spacing,
    )
    from monai.utils import set_determinism

    set_determinism(seed=42)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Handle "no prompts" case: segment all classes
    if label_prompt is None and points is None:
        everything_labels = list(
            set([i + 1 for i in range(132)]) - set([2, 16, 18, 20, 21, 23, 24, 25, 26])
        )
        label_prompt = everything_labels
        print(
            f"No prompt provided. Using everything_labels for {len(everything_labels)} classes."
        )

    if points is not None and point_labels is None:
        raise ValueError("point_labels must be provided when points are specified")

    # 3. Download model bundle if needed
    if bundle_path is None:
        import tempfile

        if model_cache_dir is None:
            model_cache_dir = tempfile.mkdtemp()
        try:
            download(name="vista3d", bundle_dir=model_cache_dir, source="monaihosting")
        except Exception:
            download(name="vista3d", bundle_dir=model_cache_dir, source="github")
        bundle_path = f"{model_cache_dir}/vista3d"

    # 4. ITK->MetaTensor (in memory)
    meta_tensor = itk_image_to_metatensor(
        itk_image, channel_dim=None, dtype=torch.float32
    )

    # 5. Preprocessing pipeline
    processed = meta_tensor
    processed = EnsureChannelFirst(channel_dim=None)(processed)
    processed = EnsureType(dtype=torch.float32)(processed)
    processed = Spacing(pixdim=[1.5, 1.5, 1.5], mode="bilinear")(processed)
    processed = ScaleIntensityRange(
        a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
    )(processed)
    processed = CropForeground()(processed)

    # Save the MONAI affine now (Spacing + CropForeground have updated it).
    # We need it later to place the label map in the processed world space before
    # resampling back to the original ITK grid.
    processed_affine = (
        processed.meta["affine"].numpy()
        if hasattr(processed, "meta") and "affine" in processed.meta
        else None
    )

    # 6. Load VISTA3D
    model = vista3d132(encoder_embed_dim=48, in_channels=1)
    model_path = f"{bundle_path}/models/model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    # 7. Prepare input tensor
    input_tensor = processed
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(np.asarray(input_tensor), dtype=torch.float32)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # 8. Prepare model inputs
    model_inputs = {"image": input_tensor}
    if label_prompt is not None:
        label_prompt_tensor = torch.tensor(
            label_prompt, dtype=torch.long, device=device
        )
        model_inputs["label_prompt"] = label_prompt_tensor
        print("label_prompt_tensor shape", label_prompt_tensor.shape)
    if points is not None:
        point_coords = torch.tensor(
            points, dtype=torch.float32, device=device
        ).unsqueeze(0)
        point_labels_tensor = torch.tensor(
            point_labels, dtype=torch.float32, device=device
        ).unsqueeze(0)
        model_inputs["points"] = point_coords
        model_inputs["point_labels"] = point_labels_tensor
        print("point_coords shape", point_coords.shape)

    # 9. Sliding window inference for large images
    def predictor_fn(x):
        args = {k: v for k, v in model_inputs.items() if k != "image"}
        print(x.shape)
        return model(x, **args)

    with torch.no_grad():
        if any(dim > 128 for dim in input_tensor.shape[2:]):
            print("Sliding window inference")
            output = sliding_window_inference(
                input_tensor,
                roi_size=[128, 128, 128],
                sw_batch_size=1,
                predictor=predictor_fn,
                overlap=0.5,
                mode="gaussian",
                device=device,
            )
        else:
            print("Single window inference")
            output = model(
                input_tensor, **{k: v for k, v in model_inputs.items() if k != "image"}
            )

    print("output shape", output.shape)
    # 10. Postprocess: multi-class to label map
    output = output.cpu()
    if hasattr(output, "detach"):
        output = output.detach()
    if isinstance(output, dict):
        if "pred" in output:
            output = output["pred"]
        else:
            output = list(output.values())[0]

    if output.shape[1] > 1:
        label_map = torch.argmax(output, dim=1).squeeze(0).numpy().astype(np.uint16)
    else:
        label_map = (output > 0.5).squeeze(0).cpu().numpy().astype(np.uint8)

    # MONAI outputs are in (D, H, W) = (z, y, x) — matches ITK's GetImageFromArray
    # convention, so no transpose is needed.
    label_map_for_itk = label_map

    # Build an ITK image in the processed (1.5 mm, cropped) world space.
    output_itk = itk.GetImageFromArray(label_map_for_itk)
    if processed_affine is not None:
        # Extract spacing and origin from the MONAI affine matrix.
        # Columns norms of the 3×3 rotation-scale block give voxel spacing.
        spacing_processed = np.sqrt(
            (processed_affine[:3, :3] ** 2).sum(axis=0)
        ).tolist()
        origin_processed = processed_affine[:3, 3].tolist()
        output_itk.SetSpacing(spacing_processed)
        output_itk.SetOrigin(origin_processed)
        output_itk.SetDirection(itk_image.GetDirection())

        # Resample the label map back to the original input image grid using
        # nearest-neighbour interpolation (preserves discrete label values).
        resampler = itk.ResampleImageFilter.New(output_itk)
        resampler.SetReferenceImage(itk_image)
        resampler.SetUseReferenceImage(True)
        resampler.SetInterpolator(
            itk.NearestNeighborInterpolateImageFunction.New(output_itk)
        )
        resampler.SetDefaultPixelValue(0)
        resampler.Update()
        output_itk = resampler.GetOutput()
    else:
        # Fallback: copy input metadata (spatial alignment may be approximate).
        output_itk.SetSpacing(itk_image.GetSpacing())
        output_itk.SetOrigin(itk_image.GetOrigin())
        output_itk.SetDirection(itk_image.GetDirection())

    return output_itk


# %%
# Load an ITK image
image = itk.imread("results/slice_fixed.mha")

spleen_segmentation = vista3d_inference_from_itk(
    image, model_cache_dir="./network_weights"
)

itk.imwrite(spleen_segmentation, "results/slice_fixed.all_mask_vista3d_inMem.mha")
