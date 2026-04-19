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

intra_donut_spots_sensitivity = model.spots_sensitivity(
    intra_donut_field,
    steps=sf.from_f(steps),
    nrad=20,
    include_chip_heights=False,
    tqdm=tqdm,
    focus="intra",
)
extra_donut_spots_sensitivity = model.spots_sensitivity(
    extra_donut_field,
    steps=sf.from_f(steps),
    nrad=20,
    include_chip_heights=False,
    tqdm=tqdm,
    focus="extra",
)

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
science_vignetted = science_spots_sensitivity.nominal.vignetted
x0[science_vignetted] = np.nan
y0[science_vignetted] = np.nan
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

donut_intra_x0 = intra_donut_spots_sensitivity.nominal.dx.to_value("micron")
donut_intra_y0 = intra_donut_spots_sensitivity.nominal.dy.to_value("micron")
donut_intra_vignetted = intra_donut_spots_sensitivity.nominal.vignetted
donut_intra_x0[donut_intra_vignetted] = np.nan
donut_intra_y0[donut_intra_vignetted] = np.nan
donut_intra_dx = intra_donut_spots_sensitivity.gradient.dx.to_value("micron")
donut_intra_dy = intra_donut_spots_sensitivity.gradient.dy.to_value("micron")

donut_extra_x0 = extra_donut_spots_sensitivity.nominal.dx.to_value("micron")
donut_extra_y0 = extra_donut_spots_sensitivity.nominal.dy.to_value("micron")
donut_extra_vignetted = extra_donut_spots_sensitivity.nominal.vignetted
donut_extra_x0[donut_extra_vignetted] = np.nan
donut_extra_y0[donut_extra_vignetted] = np.nan
donut_extra_dx = extra_donut_spots_sensitivity.gradient.dx.to_value("micron")
donut_extra_dy = extra_donut_spots_sensitivity.gradient.dy.to_value("micron")

donut_x0 = np.concatenate([donut_intra_x0, donut_extra_x0], axis=0)
donut_y0 = np.concatenate([donut_intra_y0, donut_extra_y0], axis=0)
donut_dx = np.concatenate([donut_intra_dx, donut_extra_dx], axis=1)
donut_dy = np.concatenate([donut_intra_dy, donut_extra_dy], axis=1)

donut_nfield, donut_nray = donut_x0.shape
donut_N = donut_nfield * donut_nray

intra_ocs = intra_donut_field.angle.ocs
extra_ocs = extra_donut_field.angle.ocs
donut_field_thx = np.concatenate([
    intra_ocs.x.to_value("deg"),
    extra_ocs.x.to_value("deg"),
])
donut_field_thy = np.concatenate([
    intra_ocs.y.to_value("deg"),
    extra_ocs.y.to_value("deg"),
])

donut_x0_flat = donut_x0.ravel().astype(np.float32)
donut_y0_flat = donut_y0.ravel().astype(np.float32)
donut_Sx = donut_dx.reshape(K_dof, donut_N).astype(np.float32)
donut_Sy = donut_dy.reshape(K_dof, donut_N).astype(np.float32)
donut_field_x_flat = np.repeat(donut_field_thx, donut_nray).astype(np.float32)
donut_field_y_flat = np.repeat(donut_field_thy, donut_nray).astype(np.float32)

# Pack binary: spot_xy (micron, interleaved), Sx, Sy, field_xy (deg, interleaved)
spot_xy = np.empty(2 * N, dtype=np.float32)
spot_xy[0::2] = x0_flat
spot_xy[1::2] = y0_flat

field_xy = np.empty(2 * N, dtype=np.float32)
field_xy[0::2] = field_x_flat
field_xy[1::2] = field_y_flat

sx_flat = Sx.ravel(order="C")
sy_flat = Sy.ravel(order="C")

donut_spot_xy = np.empty(2 * donut_N, dtype=np.float32)
donut_spot_xy[0::2] = donut_x0_flat
donut_spot_xy[1::2] = donut_y0_flat

donut_field_xy = np.empty(2 * donut_N, dtype=np.float32)
donut_field_xy[0::2] = donut_field_x_flat
donut_field_xy[1::2] = donut_field_y_flat

donut_sx_flat = donut_Sx.ravel(order="C")
donut_sy_flat = donut_Sy.ravel(order="C")

spot_offset = 0
sx_offset = spot_offset + spot_xy.size
sy_offset = sx_offset + sx_flat.size
field_offset = sy_offset + sy_flat.size

donut_spot_offset = field_offset + field_xy.size
donut_sx_offset = donut_spot_offset + donut_spot_xy.size
donut_sy_offset = donut_sx_offset + donut_sx_flat.size
donut_field_offset = donut_sy_offset + donut_sy_flat.size

packed = np.concatenate([
    spot_xy,
    sx_flat,
    sy_flat,
    field_xy,
    donut_spot_xy,
    donut_sx_flat,
    donut_sy_flat,
    donut_field_xy,
])
packed.tofile("model.f32")

meta = {
    "version": 5,
    "dtype": "float32",
    "N": int(N),
    "K": int(K_dof),
    "donut_N": int(donut_N),
    "donut_nfield_intra": int(donut_intra_x0.shape[0]),
    "donut_nfield_extra": int(donut_extra_x0.shape[0]),
    "donut_nray": int(donut_nray),
    "donut_clock_angle_deg": 10.0,
    "donut_clock_radius_scale": 1.4,
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
        "donut_spot_xy": {
            "offset_f32": int(donut_spot_offset),
            "length_f32": int(donut_spot_xy.size),
            "encoding": "xy_interleaved",
        },
        "donut_Sx": {
            "offset_f32": int(donut_sx_offset),
            "length_f32": int(donut_sx_flat.size),
            "shape": [int(K_dof), int(donut_N)],
            "order": "C",
        },
        "donut_Sy": {
            "offset_f32": int(donut_sy_offset),
            "length_f32": int(donut_sy_flat.size),
            "shape": [int(K_dof), int(donut_N)],
            "order": "C",
        },
        "donut_field_xy": {
            "offset_f32": int(donut_field_offset),
            "length_f32": int(donut_field_xy.size),
            "encoding": "xy_interleaved",
        },
    },
}

with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

