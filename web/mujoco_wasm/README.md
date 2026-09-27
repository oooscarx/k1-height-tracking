# K1 MuJoCo WASM demo

[Live demo](https://oooscarx.github.io/k1-height-tracking/)

This app runs the exported K1 height-tracking actor entirely in the browser:

- MuJoCo 3.14 WASM at 200 Hz
- ONNX Runtime Web actor inference at 50 Hz
- the K1 22-DoF MJCF and meshes from `third_party/booster_assets`
- the exact 365-dimensional, five-frame observation contract used in training

```bash
cd web/mujoco_wasm
npm ci
npm run dev
```

The asset sync runs automatically before development and production builds. The
committed policy manifest records the source checkpoint and all deployment
constants used by the browser controller.

The browser starts with external assistance disabled. The optional demo-assist
toggle applies a soft base-height/upright constraint and is retained only as a
diagnostic aid; it is not part of the exported actor or training result.

The demo uses checkpoint `model_307500.pt`. Its SHA-256 hashes, joint order,
action limits, gains, observation layout, and control rates are recorded in
`public/policy/k1_height_policy.json`.

Controls:

- Drag to orbit and scroll to zoom.
- Use the slider or `W`/`S` to change the target height.
- Press `1`/`2`/`3` for low, middle, and standing commands.
- Use the arrow keys or the lightning button to apply a push.
- Press `Space` to pause and `R` to reset.

Implementation details follow MuJoCo's official mesh preprocessing and
`mjvGeom.dataid` conventions:

- https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-mesh
- https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjvgeom
