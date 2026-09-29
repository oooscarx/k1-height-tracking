import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import loadMujoco from "@mujoco/mujoco";
import * as ort from "onnxruntime-web";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicDir = path.join(root, "public");
const manifestPath = process.env.K1_DIAGNOSTIC_MANIFEST ?? path.join(publicDir, "policy/k1_height_policy.json");
const policyPath = process.env.K1_DIAGNOSTIC_POLICY ?? path.join(publicDir, "policy/k1_height_policy.onnx");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const modelFiles = JSON.parse(fs.readFileSync(path.join(publicDir, "models/K1/files.json"), "utf8"));

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(
  fs.readFileSync(policyPath),
  {executionProviders: ["wasm"]},
);
const mujoco = await loadMujoco({
  wasmBinary: fs.readFileSync(path.join(root, "node_modules/@mujoco/mujoco/mujoco.wasm")),
});
mujoco.FS.mkdir("/working");
mujoco.FS.mkdir("/working/meshes");
for (const relativePath of modelFiles) {
  mujoco.FS.writeFile(
    `/working/${relativePath}`,
    fs.readFileSync(path.join(publicDir, "models/K1", relativePath)),
  );
}

class HistoryBank {
  constructor(frame) {
    this.values = new Map();
    for (const term of manifest.observation_layout) {
      this.values.set(term.name, Array.from({length: manifest.history_length}, () => [...frame[term.name]]));
    }
  }

  push(frame) {
    for (const term of manifest.observation_layout) {
      const history = this.values.get(term.name);
      history.shift();
      history.push([...frame[term.name]]);
    }
  }

  flatten() {
    const output = new Float32Array(manifest.input_size);
    let cursor = 0;
    for (const term of manifest.observation_layout) {
      for (const sample of this.values.get(term.name)) {
        for (const value of sample) output[cursor++] = value;
      }
    }
    return output;
  }
}

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

function observe(data, action, commandHeight) {
  const [w, x, y, z] = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
  const jointPosition = [];
  const jointVelocity = [];
  for (let index = 0; index < manifest.output_size; index++) {
    jointPosition.push(data.qpos[7 + index] - manifest.action_center[index]);
    jointVelocity.push(data.qvel[6 + index] * 0.05);
  }
  return {
    base_ang_vel: [data.sensordata[4] * 0.2, data.sensordata[5] * 0.2, data.sensordata[6] * 0.2],
    projected_gravity: [
      -2 * (x * z - w * y),
      -2 * (y * z + w * x),
      -(1 - 2 * (x * x + y * y)),
    ],
    joint_pos_rel: jointPosition,
    joint_vel_rel: jointVelocity,
    last_action: Array.from(
      action,
      (value, index) => value * (manifest.last_action_scales?.[index] ?? 1),
    ),
    height_command: [commandHeight],
  };
}

function policyCenter(index, commandHeight) {
  const [minimumHeight, maximumHeight] = manifest.height_posture_range;
  const base = manifest.action_center[index];
  const low = manifest.height_posture_center?.[index];
  const high = manifest.height_posture_high_center?.[index];
  if (low !== undefined && high !== undefined) {
    const phase = posturePhase(commandHeight, minimumHeight, maximumHeight);
    const lowCenter = base + manifest.height_posture_blend * (low - base);
    return lowCenter + phase * (high - lowCenter);
  }
  if (high !== undefined) {
    const phase = Math.pow(
      clamp((commandHeight - minimumHeight) / (maximumHeight - minimumHeight), 0, 1),
      manifest.height_posture_exponent,
    );
    return base + manifest.height_posture_blend * phase * (high - base);
  }
  if (low !== undefined) {
    const phase = Math.pow(
      clamp((maximumHeight - commandHeight) / (maximumHeight - minimumHeight), 0, 1),
      manifest.height_posture_exponent,
    );
    return base + manifest.height_posture_blend * phase * (low - base);
  }
  return base;
}

function posturePhase(commandHeight, minimumHeight, maximumHeight) {
  const knots = manifest.height_posture_phase_knots;
  if (!knots || knots.length < 2) {
    return Math.pow(
      clamp((commandHeight - minimumHeight) / (maximumHeight - minimumHeight), 0, 1),
      manifest.height_posture_exponent,
    );
  }
  if (commandHeight <= knots[0][0]) return knots[0][1];
  for (let index = 1; index < knots.length; index++) {
    const [endHeight, endPhase] = knots[index];
    const [startHeight, startPhase] = knots[index - 1];
    if (commandHeight <= endHeight) {
      const blend = clamp((commandHeight - startHeight) / (endHeight - startHeight), 0, 1);
      return startPhase + blend * (endPhase - startPhase);
    }
  }
  return knots.at(-1)[1];
}

async function simulate({
  rootHeight,
  delaySteps,
  actionClip,
  commandHeights,
  holdSeconds,
  gainScale,
  groundFriction,
  pushForce,
}) {
  const model = mujoco.MjModel.mj_loadXML("/working/K1_22dof.xml");
  for (let index = 0; index < manifest.output_size; index++) {
    model.dof_armature[6 + index] = manifest.joint_armature[index];
  }
  const groundGeomId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "ground");
  const trunkBodyId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "Trunk");
  if (groundGeomId >= 0) {
    model.geom_friction[groundGeomId * 3] = groundFriction;
    model.geom_friction[groundGeomId * 3 + 1] = 0.005;
    model.geom_friction[groundGeomId * 3 + 2] = 0.0001;
  }
  const data = new mujoco.MjData(model);
  const action = new Float32Array(manifest.output_size);
  const jointTargets = Float32Array.from(manifest.action_center);
  const targetDelayHistory = [jointTargets.slice()];
  let commandHeight = 0.72;
  let targetRateLimitInitialized = false;
  let targetRateLimitSteps = 0;

  data.qpos.fill(0);
  data.qvel.fill(0);
  data.qpos[2] = rootHeight;
  data.qpos[3] = 1;
  for (let index = 0; index < manifest.output_size; index++) {
    data.qpos[7 + index] = manifest.action_center[index];
  }
  mujoco.mj_forward(model, data);
  const history = new HistoryBank(observe(data, action, commandHeight));

  const updateTargets = () => {
    const [fadeStart, fadeEnd] = manifest.height_residual_fade_range;
    const phase = clamp((fadeEnd - commandHeight) / (fadeEnd - fadeStart), 0, 1);
    const smoothPhase = phase * phase * (3 - 2 * phase);
    const amplifiedJointIndices = manifest.height_residual_amplify_joint_indices;
    const baseMaximumScale = manifest.height_residual_base_maximum_scale ??
      (amplifiedJointIndices ? 1 : (manifest.height_residual_maximum_scale ?? 1));
    const baseHeightResidualScale = manifest.height_residual_minimum_scale +
      (baseMaximumScale - manifest.height_residual_minimum_scale) * smoothPhase;
    const amplifiedHeightResidualScale = manifest.height_residual_minimum_scale +
      ((manifest.height_residual_maximum_scale ?? 1) - manifest.height_residual_minimum_scale) *
        smoothPhase;
    let handedOffAmplifiedScale = amplifiedHeightResidualScale;
    if (manifest.height_residual_handoff_range) {
      const [handoffStart, handoffEnd] = manifest.height_residual_handoff_range;
      const handoffPhase = clamp((handoffEnd - commandHeight) / (handoffEnd - handoffStart), 0, 1);
      const handoffSmooth = handoffPhase * handoffPhase * (3 - 2 * handoffPhase);
      handedOffAmplifiedScale *=
        1 + (manifest.height_residual_handoff_minimum_scale - 1) * handoffSmooth;
    }
    let deepScale = 1;
    if (manifest.height_residual_deep_handoff_range) {
      const [deepStart, deepEnd] = manifest.height_residual_deep_handoff_range;
      const deepPhase = clamp((deepEnd - commandHeight) / (deepEnd - deepStart), 0, 1);
      const deepSmooth = deepPhase * deepPhase * (3 - 2 * deepPhase);
      deepScale *= 1 + (manifest.height_residual_deep_handoff_minimum_scale - 1) * deepSmooth;
    }
    const faded = new Set(manifest.height_residual_fade_joint_indices);
    const amplified = new Set(amplifiedJointIndices ?? []);
    const deep = new Set(manifest.height_residual_deep_handoff_joint_indices ?? []);
    for (let index = 0; index < manifest.output_size; index++) {
      const delta = clamp(
        manifest.action_scale[index] * action[index],
        manifest.position_delta_clip[0],
        manifest.position_delta_clip[1],
      );
      let residualScale = amplified.has(index)
        ? handedOffAmplifiedScale
        : faded.has(index)
          ? baseHeightResidualScale
          : 1;
      if (deep.has(index)) residualScale *= deepScale;
      const safeMinimum = manifest.position_minimum[index] + manifest.position_target_margin;
      const safeMaximum = manifest.position_maximum[index] - manifest.position_target_margin;
      let requested = clamp(policyCenter(index, commandHeight) + residualScale * delta, safeMinimum, safeMaximum);
      const current = data.qpos[7 + index];
      const velocity = data.qvel[6 + index];
      const predicted = current + velocity * manifest.position_braking_horizon_s;
      if ((velocity > 0 && predicted > safeMaximum) || (velocity < 0 && predicted < safeMinimum)) {
        requested = clamp(current - velocity * manifest.position_braking_horizon_s, safeMinimum, safeMaximum);
      }
      const maximumStep = manifest.position_target_velocity_limit[index] / manifest.policy_rate_hz;
      const initialTarget = manifest.position_target_velocity_limit_initialize_from_target
        ? requested
        : current;
      const previousTarget = targetRateLimitInitialized ? jointTargets[index] : initialTarget;
      const limitedTarget = previousTarget + clamp(requested - previousTarget, -maximumStep, maximumStep);
      const warmupSteps = manifest.position_target_velocity_limit_warmup_steps ?? 0;
      const rateLimited = targetRateLimitSteps < warmupSteps ? requested : limitedTarget;
      const errorLimit = manifest.torque_limit[index] / manifest.stiffness[index];
      jointTargets[index] = current + clamp(rateLimited - current, -errorLimit, errorLimit);
    }
    targetRateLimitInitialized = true;
    targetRateLimitSteps++;
  };

  const runPolicy = async () => {
    history.push(observe(data, action, commandHeight));
    const input = history.flatten();
    const output = await session.run({
      [manifest.input_name]: new ort.Tensor("float32", input, [1, manifest.input_size]),
    });
    const values = output[manifest.output_name].data;
    for (let index = 0; index < manifest.output_size; index++) {
      action[index] = clamp(values[index] ?? 0, -actionClip, actionClip);
    }
    updateTargets();
  };

  const applyController = () => {
    data.qfrc_applied.fill(0);
    for (let index = 0; index < manifest.output_size; index++) {
      const current = data.qpos[7 + index];
      const historyIndex = Math.max(0, targetDelayHistory.length - 1 - delaySteps);
      const positionError = targetDelayHistory[historyIndex][index] - current;
      const velocity = data.qvel[6 + index];
      const rawTorque = gainScale * (
        manifest.stiffness[index] * positionError - manifest.damping[index] * velocity
      );
      const speed = Math.abs(velocity);
      const effortLimit = manifest.motor_effort_limit[index];
      const maximumVelocity = manifest.motor_velocity_limit[index];
      const kneeVelocity = Math.min(manifest.motor_knee_velocity[index], maximumVelocity);
      const speedLimit = speed <= kneeVelocity || kneeVelocity >= maximumVelocity
        ? effortLimit
        : clamp(
            effortLimit * (maximumVelocity - speed) / (maximumVelocity - kneeVelocity),
            0,
            effortLimit,
          );
      data.qfrc_applied[6 + index] = clamp(rawTorque, -speedLimit, speedLimit);
    }
  };

  await runPolicy();
  const trace = [];
  const physicsSteps = Math.round(commandHeights.length * holdSeconds * manifest.simulation_rate_hz);
  const policyDivider = Math.round(manifest.simulation_rate_hz / manifest.policy_rate_hz);
  const integrationSubsteps = Math.max(
    1,
    Math.round((1 / manifest.simulation_rate_hz) / Number(model.opt.timestep)),
  );
  for (let step = 1; step <= physicsSteps; step++) {
    const elapsedSeconds = (step - 1) / manifest.simulation_rate_hz;
    const segment = Math.min(Math.floor(elapsedSeconds / holdSeconds), commandHeights.length - 1);
    const requestedHeight = commandHeights[segment];
    const maximumCommandStep = 0.08 / manifest.simulation_rate_hz;
    commandHeight += clamp(requestedHeight - commandHeight, -maximumCommandStep, maximumCommandStep);
    targetDelayHistory.push(jointTargets.slice());
    while (targetDelayHistory.length > delaySteps + 1) targetDelayHistory.shift();
    applyController();
    data.xfrc_applied.fill(0);
    if (pushForce > 0 && trunkBodyId >= 0 && elapsedSeconds >= 5) {
      const pushIndex = Math.floor(elapsedSeconds / 5) - 1;
      const pushPhase = elapsedSeconds % 5;
      if (pushPhase < 0.12) {
        const angle = pushIndex * Math.PI / 2;
        const wrenchOffset = trunkBodyId * 6;
        data.xfrc_applied[wrenchOffset] = pushForce * Math.cos(angle);
        data.xfrc_applied[wrenchOffset + 1] = pushForce * Math.sin(angle);
      }
    }
    for (let substep = 0; substep < integrationSubsteps; substep++) mujoco.mj_step(model, data);
    if (step % policyDivider === 0) {
      await runPolicy();
      trace.push({
        time_s: Number(data.time),
        requested_height: requestedHeight,
        command_height: commandHeight,
        measured_height: data.qpos[2] + manifest.tracked_point_offset_m,
        root_position: Array.from(data.qpos.slice(0, 3)),
        root_quaternion: Array.from(data.qpos.slice(3, 7)),
        joint_position: Array.from(data.qpos.slice(7, 29)),
        joint_velocity: Array.from(data.qvel.slice(6, 28)),
        action: Array.from(action),
        target: Array.from(jointTargets),
        torque: Array.from(data.qfrc_applied.slice(6, 28)),
      });
    }
  }
  const last = trace.at(-1);
  const [w, x, y] = last.root_quaternion;
  const upZ = 1 - 2 * (x * x + y * y);
  const planarDistance = Math.hypot(last.root_position[0], last.root_position[1]);
  const settled = trace.filter((sample) => {
    const withinSegment = sample.time_s % holdSeconds;
    return withinSegment >= Math.max(holdSeconds - 1, 0);
  });
  const errors = settled.map((sample) => Math.abs(sample.measured_height - sample.command_height));
  data.delete();
  model.delete();
  return {
    root_height: rootHeight,
    delay_steps: delaySteps,
    action_clip: actionClip,
    gain_scale: gainScale,
    ground_friction: groundFriction,
    push_force_n: pushForce,
    command_heights: commandHeights,
    final_root_height: last.root_position[2],
    final_up_z: upZ,
    final_planar_distance: planarDistance,
    minimum_root_height: Math.min(...trace.map((sample) => sample.root_position[2])),
    maximum_planar_distance: Math.max(...trace.map((sample) => Math.hypot(sample.root_position[0], sample.root_position[1]))),
    maximum_tilt_rad: Math.max(...trace.map((sample) => Math.acos(clamp(1 - 2 * (sample.root_quaternion[1] ** 2 + sample.root_quaternion[2] ** 2), -1, 1)))),
    settled_mean_abs_error_m: errors.reduce((sum, value) => sum + value, 0) / errors.length,
    settled_max_abs_error_m: Math.max(...errors),
    trace,
  };
}

const parseValues = (name, fallback) => process.env[name]
  ? process.env[name].split(",").map(Number)
  : fallback;
const commandHeights = parseValues("K1_DIAGNOSTIC_HEIGHTS", [0.72, 0.66, 0.62, 0.59, 0.57, 0.585, 0.62, 0.72]);
const holdSeconds = Number(process.env.K1_DIAGNOSTIC_HOLD_S ?? 3);
const results = [];
for (const rootHeight of parseValues("K1_DIAGNOSTIC_ROOT_HEIGHTS", [0.548])) {
  for (const delaySteps of parseValues("K1_DIAGNOSTIC_DELAYS", [5])) {
    for (const actionClip of parseValues("K1_DIAGNOSTIC_ACTION_CLIPS", [manifest.action_clip[1]])) {
      for (const gainScale of parseValues("K1_DIAGNOSTIC_GAIN_SCALES", [1])) {
        for (const groundFriction of parseValues("K1_DIAGNOSTIC_GROUND_FRICTIONS", [1])) {
          for (const pushForce of parseValues("K1_DIAGNOSTIC_PUSH_FORCES", [0])) {
            results.push(await simulate({
              rootHeight,
              delaySteps,
              actionClip,
              commandHeights,
              holdSeconds,
              gainScale,
              groundFriction,
              pushForce,
            }));
          }
        }
      }
    }
  }
}
const summary = results.map(({trace, ...result}) => result);
console.log(JSON.stringify(summary, null, 2));
if (process.env.K1_DIAGNOSTIC_OUTPUT) {
  fs.writeFileSync(process.env.K1_DIAGNOSTIC_OUTPUT, JSON.stringify({summary, results}, null, 2));
}
