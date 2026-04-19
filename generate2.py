import os
from pathlib import Path

import json
import astropy.units as u
import batoid
import numpy as np
import yaml
from astropy.coordinates import Angle
from batoid_rubin import LSSTBuilder
from lsst.obs.lsst import LsstCam
from StarSharp import RaytracedOpticalModel, StateFactory
from tqdm import tqdm

camera = LsstCam().getCamera()
fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
builder = LSSTBuilder(
    fiducial,
    dof_coord_system="OCS",
    flip_m2_bending_modes=False,
    dof_angle_units="degree",
    inex_optic="Detector",
)
model = RaytracedOpticalModel(
    builder,
    rtp=Angle("0 deg"),
    wavelength=620 * u.nm,
    camera=camera,
)

# Load sensitivity and normalization from ts_config_mttcs / ts_ofc
range_weight_path = (
    Path(os.environ["TS_OFC_DIR"]) / "policy" / "normalization_weights" / "range.yaml"
)
with open(range_weight_path, "r") as f:
    range_weights = np.array(yaml.safe_load(f))
fwhm_weight_path = (
    Path(os.environ["TS_OFC_DIR"]) / "policy" / "normalization_weights" / "fwhm.yaml"
)
with open(fwhm_weight_path, "r") as f:
    fwhm_weights = np.array(yaml.safe_load(f))
sensitivity_path = (
    Path(os.environ["TS_OFC_DIR"])
    / "policy"
    / "sensitivity_matrix"
    / "lsst_sensitivity_dz_31_29_50.yaml"
)
with open(sensitivity_path, "r") as f:
    sensitivity = np.array(yaml.safe_load(f))

sf = StateFactory(sensitivity, norm=range_weights * fwhm_weights)

science_field = model.make_hex_field(outer=1.75 * u.deg, nrad=5)
intra_donut_field = model.make_ccd_field(nx=1, types="ITL_WF", detnums=[192, 196, 200, 204])
extra_donut_field = model.make_ccd_field(nx=1, types="ITL_WF", detnums=[191, 195, 199, 203])

# Hard-coding the step sizes here for now.
steps = [
    10.0,
    100.0, 100.0,
    0.001, 0.001,
    10.0,
    200.0, 200.0,
    0.001, 0.001,
] + [0.1] * 40

science_spots_sensitivity = model.spots_sensitivity(
    science_field,
    steps=sf.from_f(steps),
    nrad=12,
    include_chip_heights=False,
    tqdm=tqdm,
)

# intra_donut_spots_sensitivity = model.spots_sensitivity(
#     intra_donut_field,
#     steps=sf.from_f(steps),
#     nrad=20,
#     include_chip_heights=False,
#     tqdm=tqdm,
#     focus="intra",
# )
# extra_donut_spots_sensitivity = model.spots_sensitivity(
#     extra_donut_field,
#     steps=sf.from_f(steps),
#     nrad=20,
#     include_chip_heights=False,
#     tqdm=tqdm,
#     focus="extra",
# )

# intra_donut_zernikes_sensitivity = model.zernikes_sensitivity(
#     intra_donut_field,
#     steps=sf.from_f(steps),
#     include_chip_heights=False,
#     tqdm=tqdm,
# )
# extra_donut_zernikes_sensitivity = model.zernikes_sensitivity(
#     extra_donut_field,
#     steps=sf.from_f(steps),
#     include_chip_heights=False,
#     tqdm=tqdm,
# )

x0 = science_spots_sensitivity.nominal.dx.to_value("micron")
y0 = science_spots_sensitivity.nominal.dy.to_value("micron")
dx = science_spots_sensitivity.gradient.dx.to_value("micron")
dy = science_spots_sensitivity.gradient.dy.to_value("micron")

nfield, nray = x0.shape
K_dof = dx.shape[0]
N = nfield * nray

# Field positions in degrees (OCS angles from science_field)
field_thx = science_field.x.to_value("deg")
field_thy = science_field.y.to_value("deg")

# Flatten to N = nfield * nray points (C order)
x0_flat = x0.ravel().astype(np.float32)
y0_flat = y0.ravel().astype(np.float32)
Sx = dx.reshape(K_dof, N).astype(np.float32)
Sy = dy.reshape(K_dof, N).astype(np.float32)

# Per-ray field offsets: each field position repeated nray times
field_x_flat = np.repeat(field_thx, nray).astype(np.float32)
field_y_flat = np.repeat(field_thy, nray).astype(np.float32)

# Pack binary: spot_xy (micron, interleaved), Sx, Sy, field_xy (deg, interleaved)
spot_xy = np.empty(2 * N, dtype=np.float32)
spot_xy[0::2] = x0_flat
spot_xy[1::2] = y0_flat

field_xy = np.empty(2 * N, dtype=np.float32)
field_xy[0::2] = field_x_flat
field_xy[1::2] = field_y_flat

sx_flat = Sx.ravel(order="C")
sy_flat = Sy.ravel(order="C")

spot_offset = 0
sx_offset = spot_offset + spot_xy.size
sy_offset = sx_offset + sx_flat.size
field_offset = sy_offset + sy_flat.size

packed = np.concatenate([spot_xy, sx_flat, sy_flat, field_xy])
packed.tofile("model.f32")

meta = {
    "version": 4,
    "dtype": "float32",
    "N": int(N),
    "K": int(K_dof),
    "layout": {
        "spot_xy": {
            "offset_f32": int(spot_offset),
            "length_f32": int(spot_xy.size),
            "encoding": "xy_interleaved",
        },
        "Sx": {
            "offset_f32": int(sx_offset),
            "length_f32": int(sx_flat.size),
            "shape": [int(K_dof), int(N)],
            "order": "C",
        },
        "Sy": {
            "offset_f32": int(sy_offset),
            "length_f32": int(sy_flat.size),
            "shape": [int(K_dof), int(N)],
            "order": "C",
        },
        "field_xy": {
            "offset_f32": int(field_offset),
            "length_f32": int(field_xy.size),
            "encoding": "xy_interleaved",
        },
    },
}

with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

