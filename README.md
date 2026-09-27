# Booster K1 Height Tracking

K1 reproduction of NVIDIA WBC-AGILE's `HeightTracking-G1-v0` sit-down / stand-up
task, based on WBC-AGILE commit `7259792cf10803aab814d101134d493d24c8f22f`.

The registered task is `Booster-K1-Height-Tracking-v0`. It tracks a continuously
resampled terrain-relative trunk-height command, including negative commands for
relaxed lying, intermediate crouch/sit heights, and full standing. This is not an
arbitrary fall-recovery task and uses no BC, AMP, motion clip, or locomotion policy.

## What Matches WBC-AGILE

- 200 Hz simulation and 50 Hz policy rate (`decimation=4`)
- 15 second episodes, 4,096 environments, and 24 steps per PPO rollout
- 1-7 second command resampling with 30% standing, 20% lying, and 50% uniform heights
- three height-tracking reward scales and the full posture/contact/impact reward set
- 90% body-weight lift assistance with performance-driven decay
- eight-level boxes, rough-noise, and wave terrain curriculum
- physical randomization, pushes, external wrenches, PPO symmetry, L2C2, and return normalization
- 100,000 iterations and checkpoints every 250 iterations

The robot-specific adaptation uses K1's 22-joint deployment order, K1 body names,
hardware gains, logical limits, torque-speed limits, target braking, and physical
parallel-ankle feasibility projection. Initial states are 50% standing and 50%
structured supine states. WBC-AGILE's later arbitrary-ragdoll curriculum is
intentionally disabled because this project is height tracking, not fall recovery.

## Repository Map

- `source/booster_train/.../height_tracking`: task, rewards, curricula, randomization, and K1 configuration
- `scripts/rsl_rl`: training, monitoring, playback, and continuation entry points
- `scripts/export_height_actor_onnx.py`: deterministic actor and deployment-manifest export
- `deployment/height_command.py`: deployment-side command utilities
- `web/mujoco_wasm`: standalone MuJoCo WASM and ONNX Runtime Web demo
- `tests`: focused curriculum, reward, checkpoint, monitoring, and web-control coverage

## Environment

The training checkout on Bravo is:

```text
/home/bravo/xsb/k1-height-tracking
```

The Python environment is managed by `uv` and stored in `.venv`. For a clean clone:

```bash
uv sync --frozen
```

The K1 assets and the WBC-AGILE `rsl_rl` fork are vendored under `third_party`
because both contain project-specific changes required by the trained policy.

When reusing a moved `.venv`, refresh the editable local packages without network access:

```bash
uv pip install --python .venv/bin/python3 --no-deps --no-build-isolation \
  -e third_party/booster_assets \
  -e third_party/wbc_agile_rsl_rl \
  -e source/booster_train
```

## Train

One-iteration smoke test:

```bash
cd ~/xsb/k1-height-tracking
NUM_ENVS=16 MAX_ITERATIONS=1 ./scripts/rsl_rl/train_k1_height_tracking.sh
```

Start the exact 4,096-environment baseline in tmux:

```bash
cd ~/xsb/k1-height-tracking
./scripts/rsl_rl/start_k1_height_tracking.sh
```

Attach to training or read the log:

```bash
tmux attach -t k1-height-training
tail -f ~/xsb/k1-height-tracking/logs/height_tracking_train.console.log
```

Resume a checkpoint:

```bash
./scripts/rsl_rl/train_k1_height_tracking.sh \
  --resume --resume_path /absolute/path/to/model_<iteration>.pt
```

Runs and checkpoints are written under:

```text
logs/rsl_rl/height_tracking_k1/<timestamp>_height_tracking_k1/
```

## Sparse Monitoring

`start_k1_height_tracking.sh` starts a second tmux session that records one compact
JSON object every 15 minutes. It includes iteration, reward, height error, lift
scale, terrain level, invalid-state rate, PPO losses, policy standard deviation,
and FPS.

```bash
tail -f ~/xsb/k1-height-tracking/logs/height_tracking_health.jsonl
tmux attach -t k1-height-monitor
```

Take one snapshot manually:

```bash
.venv/bin/python3 scripts/rsl_rl/monitor_k1_height_tracking.py --once
```

## Playback

Record the policy while the command generator switches among lying, intermediate,
and standing heights:

```bash
.venv/bin/python3 scripts/rsl_rl/play.py \
  --task Booster-K1-Height-Tracking-v0 \
  --checkpoint /absolute/path/to/model_<iteration>.pt \
  --num_envs 4 --video --video_length 1500 --headless
```

The command is a one-dimensional terrain-relative trunk height. The observation
contract is five frames of base angular velocity, projected gravity, 22 joint
positions, 22 joint velocities, previous actions, and the height command. Deployment
must preserve this order and the 50 Hz policy rate.

### Browser control

Run one randomized environment with a live browser view and keyboard control:

```bash
.venv/bin/python3 scripts/rsl_rl/play.py \
  --task Booster-K1-Height-Tracking-v0 \
  --checkpoint /absolute/path/to/model_<iteration>.pt \
  --num_envs 1 --web_control --headless --wbc_lift_scale 0
```

The server binds only to `127.0.0.1:8765`. From another computer, create an SSH
tunnel and open `http://127.0.0.1:8765`:

```bash
ssh -L 8765:127.0.0.1:8765 bravo-external
```

Use `W`/`S` to change height by 2 cm, `1`/`2`/`3` for relaxed/crouch/stand,
`R` to reset, and the arrow keys to apply directional velocity disturbances.
Physical domain randomization and the task's scheduled pushes/wrenches remain enabled
unless the corresponding `--wbc_disable_*` options are explicitly passed.

## MuJoCo WASM Demo

[Open the live K1 Height Tracking Demo](https://oooscarx.github.io/k1-height-tracking/)

The browser demo loads the K1 MJCF and meshes into MuJoCo 3.14 WebAssembly and
runs the exported 365-input, 22-output actor through ONNX Runtime Web. Physics
runs at 200 Hz and policy inference at 50 Hz.

```bash
cd web/mujoco_wasm
npm ci
npm run dev
```

Open the Vite URL, then use the height slider, push button, pause, and reset
controls. The committed actor comes from `model_307500.pt`; the accompanying
manifest records checkpoint and ONNX hashes plus the exact joint, observation,
gain, limit, and timing contract. The demo starts without external assistance;
the optional `辅助` toggle is kept only for simulator diagnostics. The `扰动`
toggle randomizes contact friction, gains, and scheduled pushes for interactive
robustness checks.

To export a different checkpoint:

```bash
.venv/bin/python3 scripts/export_height_actor_onnx.py \
  /absolute/path/to/model_<iteration>.pt \
  web/mujoco_wasm/public/policy/k1_height_policy.onnx
```

### Error-focused continuation

`Booster-K1-Height-Tracking-ErrorRecovery-v0` keeps the base task's rewards,
terrain curriculum, physics randomization, and disturbances, while assigning 65%
of continuous-command samples to the measured `0.30–0.72 m` error bottleneck.
It retains negative-height replay and fallen resets to limit forgetting. Resume a
mature checkpoint with zero lift using:

```bash
scripts/rsl_rl/start_k1_height_error_recovery.sh \
  /absolute/path/to/model_<iteration>.pt
```
