# Generate initial points and sensitivity

import os
from pathlib import Path

import numpy as np
import batoid
from batoid_rubin import LSSTBuilder
import json
import yaml
from star_sharp import StarSharp, PUPIL_OUTER, PUPIL_INNER
from tqdm import tqdm

# Load normalization weights and sensitivity matrix from ts_config_mttcs
mttcs_dir = Path(os.environ["TS_CONFIG_MTTCS_DIR"])
mtaos_dir = mttcs_dir / "MTAOS/v13/ofc/"

senspath = mtaos_dir / "sensitivity_matrix" / "lsst_sensitivity_dz_31_29_50.yaml"
with open(senspath, "r") as f:
    wf_sens = np.array(yaml.safe_load(f), dtype=np.float32)
# wf_sens shape: (31, 29, 50) -> flatten field*zk dimensions -> (31*29, 50)
wf_sens = wf_sens.reshape(-1, wf_sens.shape[-1])

normpath = mtaos_dir / "normalization_weights" / "range-fwhm.yaml"
with open(normpath, "r") as f:
    norm = np.array(yaml.safe_load(f), dtype=np.float64)

ssh = StarSharp(
    "i",
    transverse_pupil_radii = 12,
    transverse_field_radii = 5,
    use_dof="0-9,10-16,30-34",
    nkeep=12,
    tqdm=tqdm
)

x0 = ssh.x0
y0 = ssh.y0
dx = ssh.dx
dy = ssh.dy

factor = 3e3
x0 *= factor
y0 *= factor

for i, (u, v) in enumerate(zip(ssh.field_u, ssh.field_v)):
    x0[i] += u
    y0[i] += v

# Now the sensitivity
dx *= factor
dy *= factor

Sx = dx
Sy = dy

# Add donuts

donut_factor = 5e2

pupil_x,pupil_y = batoid.utils.hexapolar(
    outer=PUPIL_OUTER*0.99,  # Avoid clipping the actual pupil
    inner=PUPIL_INNER*1.01,
    nrad=20,
    naz=int(2 * np.pi * PUPIL_OUTER / (PUPIL_OUTER - PUPIL_INNER) * 20),
)

intra = ssh.fiducial.withGloballyShiftedOptic("Detector", [0, 0, -1.5e-3])
extra = ssh.fiducial.withGloballyShiftedOptic("Detector", [0, 0, +1.5e-3])
builder = LSSTBuilder(ssh.fiducial)
builder_in = LSSTBuilder(intra)
builder_ex = LSSTBuilder(extra)

# Add intrafocal donuts
xd = np.empty((8, len(pupil_x)))
yd = np.empty((8, len(pupil_y)))
SXd = np.empty((8, len(pupil_x), ssh.n_dof))
SYd = np.empty((8, len(pupil_y), ssh.n_dof))

# Add donut intrinsic and sensitivity Zernikes
jmax = 28
zk0 = np.empty((4, jmax+1))
dzk = np.empty((4, jmax+1, ssh.n_dof))

bar = tqdm(total=ssh.n_dof*8, desc="Generating sensitivity")
for idof, (step, sign) in enumerate(zip(ssh._steps, ssh.dof_signs)):
    dof = np.zeros(ssh.n_dof)
    dof[idof] = step * sign
    perturbed = builder.with_aos_dof(dof).build()
    perturbed_in = builder_in.with_aos_dof(dof).build()
    perturbed_ex = builder_ex.with_aos_dof(dof).build()

    for i, corner in enumerate([(-1.25, -1.25), (-1.25, 1.25), (1.25, -1.25), (1.25, 1.25)]):
        # Intra
        rays = batoid.RayVector.fromStop(
            np.array(pupil_x),
            np.array(pupil_y),
            theta_x=np.deg2rad(corner[0]),
            theta_y=np.deg2rad(corner[1]),
            optic=intra,
            wavelength=ssh.wavelength,
        )
        frays = intra.trace(rays.copy())
        prays = perturbed_in.trace(rays.copy())
        vignetted = frays.vignetted

        meandx = np.nanmean(prays.x - frays.x)
        meandy = np.nanmean(prays.y - frays.y)

        SXd[i, :, idof] = (prays.x - frays.x - meandx) / step
        SYd[i, :, idof] = (prays.y - frays.y - meandy) / step
        SXd[i, vignetted, idof] = np.nan
        SYd[i, vignetted, idof] = np.nan

        zkf = batoid.zernikeGQ(
            ssh.fiducial,
            np.deg2rad(corner[0]), np.deg2rad(corner[1]),
            ssh.wavelength,
            rings=10, jmax=28, eps=0.612
        ) * ssh.wavelength * 1e9
        zkp = batoid.zernikeGQ(
            perturbed,
            np.deg2rad(corner[0]), np.deg2rad(corner[1]),
            ssh.wavelength,
            rings=10, jmax=28, eps=0.612
        ) * ssh.wavelength * 1e9
        dzk[i, :, idof] = (zkp - zkf) / step

        if idof == 0:
            xd[i] = np.array(frays.x)
            yd[i] = np.array(frays.y)
            xd[i][vignetted] = np.nan
            yd[i][vignetted] = np.nan
            xd[i] -= np.nanmean(xd[i])
            yd[i] -= np.nanmean(yd[i])

            xd[i] *= donut_factor
            yd[i] *= donut_factor
            dth = 10
            sdth, cdth = np.sin(np.deg2rad(dth)), np.cos(np.deg2rad(dth))

            dx = cdth * corner[0] - sdth * corner[1]
            dy = sdth * corner[0] + cdth * corner[1]
            xd[i] += dx*1.4
            yd[i] += dy*1.4

            zk0[i] = zkf

        bar.update(1)

        # Extra
        rays = batoid.RayVector.fromStop(
            np.array(pupil_x),
            np.array(pupil_y),
            theta_x=np.deg2rad(corner[0]),
            theta_y=np.deg2rad(corner[1]),
            optic=extra,
            wavelength=ssh.wavelength,
        )
        frays = extra.trace(rays.copy())
        prays = perturbed_ex.trace(rays.copy())
        vignetted = frays.vignetted

        meandx = np.nanmean(prays.x - frays.x)
        meandy = np.nanmean(prays.y - frays.y)

        SXd[i+4, :, idof] = (prays.x - frays.x - meandx) / step
        SYd[i+4, :, idof] = (prays.y - frays.y - meandy) / step
        SXd[i+4, vignetted, idof] = np.nan
        SYd[i+4, vignetted, idof] = np.nan

        if idof == 0:
            xd[i+4] = np.array(frays.x)
            yd[i+4] = np.array(frays.y)
            vignetted = frays.vignetted
            xd[i+4][vignetted] = np.nan
            yd[i+4][vignetted] = np.nan
            xd[i+4] -= np.nanmean(xd[i+4])
            yd[i+4] -= np.nanmean(yd[i+4])

            xd[i+4] *= donut_factor
            yd[i+4] *= donut_factor
            dx = cdth * corner[0] + sdth * corner[1]
            dy = -sdth * corner[0] + cdth * corner[1]
            xd[i+4] += dx*1.4
            yd[i+4] += dy*1.4

        bar.update(1)

SXd *= donut_factor
SYd *= donut_factor

x0 = x0.astype(np.float32).ravel()
y0 = y0.astype(np.float32).ravel()

xd = xd.astype(np.float32).ravel()
yd = yd.astype(np.float32).ravel()

Sx = Sx.astype(np.float32).reshape(-1, ssh.n_dof)
Sy = Sy.astype(np.float32).reshape(-1, ssh.n_dof)

SXd = SXd.astype(np.float32).reshape(-1, ssh.n_dof)
SYd = SYd.astype(np.float32).reshape(-1, ssh.n_dof)

zk0 = zk0.astype(np.float32)
dzk = dzk.astype(np.float32)

x0 = np.concatenate([x0, xd])
y0 = np.concatenate([y0, yd])

Sx = np.concatenate([Sx, SXd])
Sy = np.concatenate([Sy, SYd])

Sx = Sx.T
Sy = Sy.T

N = x0.shape[0]
K = Sx.shape[0]
ZK_CORNERS, ZK_TERMS = zk0.shape

init = np.empty(2 * N, dtype=np.float32)
init[0::2] = x0
init[1::2] = y0

sx_flat = Sx.ravel(order="C")
sy_flat = Sy.ravel(order="C")
zk0_flat = zk0.ravel(order="C")
dzk_flat = dzk.ravel(order="C")
wf_sens_flat = wf_sens.astype(np.float32).ravel(order="C")

init_offset = 0
sx_offset = init_offset + init.size
sy_offset = sx_offset + sx_flat.size
zk0_offset = sy_offset + sy_flat.size
dzk_offset = zk0_offset + zk0_flat.size
sens_offset = dzk_offset + dzk_flat.size

packed = np.concatenate([init, sx_flat, sy_flat, zk0_flat, dzk_flat, wf_sens_flat])
packed.tofile("model.f32")

WF_SENS_ROWS, WF_SENS_COLS = wf_sens.shape

meta = {
    "version": 3,
    "dtype": "float32",
    "N": int(N),
    "K": int(K),
    "norm": norm.tolist(),
    "default_use_dof": [int(v) for v in ssh.use_dof.tolist()],
    "default_nkeep": int(ssh.nkeep),
    "layout": {
        "init_xy": {"offset_f32": int(init_offset), "length_f32": int(init.size), "encoding": "xy_interleaved"},
        "Sx": {"offset_f32": int(sx_offset), "length_f32": int(sx_flat.size), "shape": [int(K), int(N)], "order": "C"},
        "Sy": {"offset_f32": int(sy_offset), "length_f32": int(sy_flat.size), "shape": [int(K), int(N)], "order": "C"},
        "zk0": {"offset_f32": int(zk0_offset), "length_f32": int(zk0_flat.size), "shape": [int(ZK_CORNERS), int(ZK_TERMS)], "order": "C"},
        "dzk": {"offset_f32": int(dzk_offset), "length_f32": int(dzk_flat.size), "shape": [int(ZK_CORNERS), int(ZK_TERMS), int(K)], "order": "C"},
        "wf_sens": {"offset_f32": int(sens_offset), "length_f32": int(wf_sens_flat.size), "shape": [int(WF_SENS_ROWS), int(WF_SENS_COLS)], "order": "C"}
    }
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
