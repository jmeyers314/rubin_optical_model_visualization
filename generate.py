# Generate initial points and sensitivity

import numpy as np
import json
from star_sharp import StarSharp
from tqdm import tqdm

ssh = StarSharp(
    "i",
    transverse_pupil_radii = 8,
    transverse_field_radii = 10,
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

x0 = x0.astype(np.float32).ravel()
y0 = y0.astype(np.float32).ravel()

# Now the sensitivity
dx *= factor
dy *= factor

Sx = dx.astype(np.float32).reshape(-1, 50).T
Sy = dy.astype(np.float32).reshape(-1, 50).T

N = x0.shape[0]
K = Sx.shape[0]

init = np.empty(2 * N, dtype=np.float32)
init[0::2] = x0
init[1::2] = y0

packed = np.concatenate([init, Sx.ravel(order="C"), Sy.ravel(order="C")])
packed.tofile("model.f32")

meta = {
    "version": 1,
    "dtype": "float32",
    "N": int(N),
    "K": int(K),
    "layout": {
        "init_xy": {"offset_f32": 0, "length_f32": int(2 * N), "encoding": "xy_interleaved"},
        "Sx": {"offset_f32": int(2 * N), "length_f32": int(K * N), "shape": [int(K), int(N)], "order": "C"},
        "Sy": {"offset_f32": int(2 * N + K * N), "length_f32": int(K * N), "shape": [int(K), int(N)], "order": "C"}
    }
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
