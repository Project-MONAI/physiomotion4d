# Scripts written by the user and run through either the GUI or GUI-less, e.g. console mode, view in the software, should always be reviewed in the
# GUI version of the software before being used for any purpose. Scripts written for previous versions of Simpleware Medical should be re-validated
# by the user before being applied with the present UI-based software interface.
# Users are also recommended to review the scripting API change log after upgrading to a newer version (available within the Scripting Help menu).


import simpleware.scripting as sw

# Obtain a reference to the application
app = sw.App.GetInstance()

input_file = None
output_dir = "."
if len(app.GetInput()) > 2:
    params = app.GetInput()
    print(params)
    input_file = "file"
    output_dir = "dir"

if not app.HasActiveDocument():
    app.ImportNifitiImage(input_file)

doc = sw.App.GetDocument()
as_cardio = doc.GetAutoSegmenters().GetASCardio()

parts = sw.HeartParts(
    as_cardio.RightAtrium,
    as_cardio.LeftAtrium,
    as_cardio.RightVentricle,
    as_cardio.LeftVentricle,
    as_cardio.Myocardium,
    as_cardio.Aorta,
)
bounds = as_cardio.CalculateHeartCTRegionOfInterest(parts)

as_cardio.ApplyHeartCTTool(bounds, parts, True)

mask_names = [
    "Aorta",
    "Left Ventricle",
    "Right Ventricle",
    "Left Atrium",
    "Right Atrium",
    "Myocardium",
    "Aorta",
]

for mask_name in mask_names:
    mask = doc.GetMaskByName(mask_name)
    fixed_name = mask_name.replace(" ", "")
    mask.MetaImageExport(
        f"C:/src/Projects/PhysioMotion/physiomotion4d/experiments/mask{fixed_name}.mhd"
    )
