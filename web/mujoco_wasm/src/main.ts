import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import {createIcons, Pause, Play, RotateCcw, Zap} from "lucide";
import * as ort from "onnxruntime-web/wasm";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";

import "./style.css";

type PolicyManifest = {
  checkpoint_iteration: number;
  input_name: string;
  output_name: string;
  input_size: number;
  output_size: number;
  history_length: number;
  observation_layout: Array<{name: string; width: number; scale: number}>;
  joint_names: string[];
  action_clip: [number, number];
  action_center: number[];
  action_scale: number[];
  position_minimum: number[];
  position_maximum: number[];
  stiffness: number[];
  damping: number[];
  torque_limit: number[];
  policy_rate_hz: number;
  simulation_rate_hz: number;
  tracked_point_offset_m: number;
  standing_root_height_m: number;
};

const BASE_URL = import.meta.env.BASE_URL;
const $ = <T extends HTMLElement>(selector: string) => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
};

const ui = {
  viewport: $("#viewport"),
  loading: $("#loading"),
  loadingStage: $("#loading-stage"),
  loadingProgress: $("#loading-progress"),
  statusDot: $("#status-dot"),
  statusText: $("#status-text"),
  actualHeight: $("#actual-height"),
  heightError: $("#height-error"),
  pushForce: $("#push-force"),
  targetOutput: $<HTMLOutputElement>("#target-output"),
  heightSlider: $<HTMLInputElement>("#height-slider"),
  stabilizer: $<HTMLInputElement>("#stabilizer"),
  randomization: $<HTMLInputElement>("#randomization"),
  forceSlider: $<HTMLInputElement>("#force-slider"),
  forceOutput: $<HTMLOutputElement>("#force-output"),
  pauseButton: $<HTMLButtonElement>("#pause-button"),
  resetButton: $<HTMLButtonElement>("#reset-button"),
  pushButton: $<HTMLButtonElement>("#push-button"),
};

createIcons({icons: {Pause, Play, RotateCcw, Zap}});

class CapsuleGeometry extends THREE.BufferGeometry {
  constructor(radius: number, length: number) {
    const path = new THREE.Path();
    path.absarc(0, -length / 2, radius, Math.PI * 1.5, 0, false);
    path.absarc(0, length / 2, radius, 0, Math.PI * 0.5, false);
    const lathe = new THREE.LatheGeometry(path.getPoints(24), 16);
    super();
    this.setIndex(lathe.getIndex());
    this.setAttribute("position", lathe.getAttribute("position"));
    this.setAttribute("normal", lathe.getAttribute("normal"));
    this.setAttribute("uv", lathe.getAttribute("uv"));
    this.rotateX(Math.PI / 2);
  }
}

class HistoryBank {
  private readonly values = new Map<string, number[][]>();

  constructor(private readonly manifest: PolicyManifest) {}

  reset(frame: Record<string, number[]>): void {
    this.values.clear();
    for (const term of this.manifest.observation_layout) {
      const initial = frame[term.name];
      if (!initial || initial.length !== term.width) {
        throw new Error(`Invalid observation term ${term.name}`);
      }
      this.values.set(term.name, Array.from({length: this.manifest.history_length}, () => [...initial]));
    }
  }

  push(frame: Record<string, number[]>): void {
    for (const term of this.manifest.observation_layout) {
      const history = this.values.get(term.name);
      const sample = frame[term.name];
      if (!history || !sample) throw new Error(`Missing observation term ${term.name}`);
      history.shift();
      history.push([...sample]);
    }
  }

  flatten(): Float32Array {
    const output = new Float32Array(this.manifest.input_size);
    let cursor = 0;
    for (const term of this.manifest.observation_layout) {
      const history = this.values.get(term.name);
      if (!history) throw new Error(`History not initialized for ${term.name}`);
      for (const sample of history) {
        for (const value of sample) output[cursor++] = value;
      }
    }
    return output;
  }
}

class K1HeightDemo {
  private mujoco: any;
  private model: any;
  private data: any;
  private mjvOption: any;
  private mjvPerturb: any;
  private mjvCamera: any;
  private mjvScene: any;
  private session!: ort.InferenceSession;
  private manifest!: PolicyManifest;
  private history!: HistoryBank;

  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(42, 1, 0.02, 50);
  private readonly renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
  private readonly controls = new OrbitControls(this.camera, this.renderer.domElement);
  private readonly meshes: THREE.Mesh[] = [];
  private readonly geometryCache = new Map<string, THREE.BufferGeometry>();

  private action = new Float32Array(22);
  private targetHeight = 0.72;
  private paused = false;
  private policyPending = false;
  private accumulator = 0;
  private lastTimestamp = performance.now();
  private physicsSteps = 0;
  private pushStepsRemaining = 0;
  private pushMagnitude = 0;
  private pushX = 0;
  private pushY = 0;
  private nextRandomPush = 2.5;
  private gainScale = 1;
  private trunkBodyId = 1;
  private groundGeomId = 0;

  async initialize(): Promise<void> {
    this.setLoading("载入 MuJoCo WebAssembly", 10);
    this.mujoco = await loadMujoco({locateFile: () => mujocoWasmUrl});

    this.setLoading("同步 K1 模型", 28);
    await this.loadModelAssets();
    this.model = this.mujoco.MjModel.mj_loadXML("/working/K1_22dof.xml");
    this.data = new this.mujoco.MjData(this.model);
    this.mjvOption = new this.mujoco.MjvOption();
    this.mjvPerturb = new this.mujoco.MjvPerturb();
    this.mjvCamera = new this.mujoco.MjvCamera();
    this.mjvScene = new this.mujoco.MjvScene(this.model, 4096);
    this.trunkBodyId = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_BODY.value, "Trunk");
    this.groundGeomId = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_GEOM.value, "ground");
    for (let index = 0; index < this.mjvOption.geomgroup.length; index++) {
      this.mjvOption.geomgroup[index] = index === 1 ? 1 : 0;
    }

    this.setLoading("载入 ONNX 策略", 64);
    this.manifest = await fetch(`${BASE_URL}policy/k1_height_policy.json`).then((response) => {
      if (!response.ok) throw new Error(`Policy manifest: ${response.status}`);
      return response.json() as Promise<PolicyManifest>;
    });
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = {wasm: new URL(ortWasmUrl, window.location.href).href};
    this.session = await ort.InferenceSession.create(`${BASE_URL}policy/k1_height_policy.onnx`, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    this.history = new HistoryBank(this.manifest);

    this.setLoading("初始化控制器", 88);
    this.setupRenderer();
    this.bindControls();
    this.reset();
    await this.runPolicy();

    this.setLoading("就绪", 100);
    ui.statusDot.classList.remove("loading");
    ui.statusText.textContent = `策略 ${this.manifest.checkpoint_iteration}`;
    window.setTimeout(() => ui.loading.classList.add("hidden"), 220);
    requestAnimationFrame((time) => this.animate(time));
  }

  private setLoading(label: string, progress: number): void {
    ui.loadingStage.textContent = label;
    ui.loadingProgress.style.width = `${progress}%`;
  }

  private async loadModelAssets(): Promise<void> {
    const files = await fetch(`${BASE_URL}models/K1/files.json`).then((response) => {
      if (!response.ok) throw new Error(`Model manifest: ${response.status}`);
      return response.json() as Promise<string[]>;
    });
    this.mujoco.FS.mkdir("/working");
    this.mujoco.FS.mkdir("/working/meshes");
    for (const [index, file] of files.entries()) {
      const response = await fetch(`${BASE_URL}models/K1/${file}`);
      if (!response.ok) throw new Error(`Model asset ${file}: ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      this.mujoco.FS.writeFile(`/working/${file}`, bytes);
      this.setLoading("同步 K1 模型", 28 + Math.round((index / files.length) * 28));
    }
  }

  private setupRenderer(): void {
    this.scene.background = new THREE.Color(0x0b1114);
    this.scene.fog = new THREE.Fog(0x0b1114, 2.4, 7);

    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    ui.viewport.appendChild(this.renderer.domElement);

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1.35, -1.55, 0.94);
    this.controls.target.set(0, 0, 0.46);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 0.65;
    this.controls.maxDistance = 4;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    const hemi = new THREE.HemisphereLight(0xdde8e2, 0x25312d, 1.9);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(-1.8, -2.2, 3.2);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -2;
    key.shadow.camera.right = 2;
    key.shadow.camera.top = 2;
    key.shadow.camera.bottom = -2;
    this.scene.add(key);

    const groundMaterial = new THREE.MeshStandardMaterial({color: 0x27312e, roughness: 0.92, metalness: 0});
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(16, 16), groundMaterial);
    ground.receiveShadow = true;
    ground.position.z = -0.003;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(8, 32, 0x53625c, 0x34413c);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.001;
    const materials = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const material of materials) {
      material.opacity = 0.38;
      material.transparent = true;
    }
    this.scene.add(grid);

    window.addEventListener("resize", () => this.resize());
    this.resize();
  }

  private resize(): void {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  private bindControls(): void {
    ui.heightSlider.addEventListener("input", () => this.setTargetHeight(Number(ui.heightSlider.value)));
    ui.forceSlider.addEventListener("input", () => {
      ui.forceOutput.value = `${Number(ui.forceSlider.value).toFixed(1)}x`;
    });
    ui.pauseButton.addEventListener("click", () => this.togglePause());
    ui.resetButton.addEventListener("click", () => this.reset());
    ui.pushButton.addEventListener("click", () => this.applyPush(Math.random() < 0.5 ? -1 : 1, 0));
    document.querySelectorAll<HTMLButtonElement>("[data-height]").forEach((button) => {
      button.addEventListener("click", () => this.setTargetHeight(Number(button.dataset.height)));
    });
    window.addEventListener("keydown", (event) => {
      if (event.target instanceof HTMLInputElement) return;
      const key = event.key.toLowerCase();
      if (key === "w") this.setTargetHeight(this.targetHeight + 0.02);
      if (key === "s") this.setTargetHeight(this.targetHeight - 0.02);
      if (key === "1") this.setTargetHeight(0.54);
      if (key === "2") this.setTargetHeight(0.63);
      if (key === "3") this.setTargetHeight(0.72);
      if (key === "r") this.reset();
      if (event.code === "Space") {
        event.preventDefault();
        this.togglePause();
      }
      if (event.key === "ArrowUp") this.applyPush(1, 0);
      if (event.key === "ArrowDown") this.applyPush(-1, 0);
      if (event.key === "ArrowLeft") this.applyPush(0, 1);
      if (event.key === "ArrowRight") this.applyPush(0, -1);
    });
  }

  private setTargetHeight(value: number): void {
    this.targetHeight = THREE.MathUtils.clamp(value, -0.5, 0.72);
    ui.heightSlider.value = this.targetHeight.toFixed(2);
    ui.targetOutput.value = `${this.targetHeight.toFixed(2)} m`;
    document.querySelectorAll<HTMLButtonElement>("[data-height]").forEach((button) => {
      button.classList.toggle("active", Math.abs(Number(button.dataset.height) - this.targetHeight) < 0.005);
    });
  }

  private togglePause(): void {
    this.paused = !this.paused;
    ui.pauseButton.innerHTML = this.paused ? '<i data-lucide="play"></i>' : '<i data-lucide="pause"></i>';
    ui.pauseButton.title = this.paused ? "继续" : "暂停";
    ui.pauseButton.setAttribute("aria-label", this.paused ? "继续" : "暂停");
    createIcons({icons: {Pause, Play}});
    ui.statusDot.classList.toggle("paused", this.paused);
    ui.statusText.textContent = this.paused ? "已暂停" : `策略 ${this.manifest.checkpoint_iteration}`;
    this.lastTimestamp = performance.now();
  }

  private reset(): void {
    this.mujoco.mj_resetData(this.model, this.data);
    this.data.qpos.fill(0);
    this.data.qvel.fill(0);
    // The MJCF foot collision boxes sit 28 mm below the Isaac/URDF reset plane.
    this.data.qpos[2] = this.manifest.standing_root_height_m + 0.028;
    this.data.qpos[3] = 1;
    for (let index = 0; index < this.manifest.output_size; index++) {
      this.data.qpos[7 + index] = this.manifest.action_center[index];
    }
    this.action.fill(0);
    this.pushStepsRemaining = 0;
    this.pushMagnitude = 0;
    this.pushX = 0;
    this.pushY = 0;
    this.physicsSteps = 0;
    this.accumulator = 0;
    this.gainScale = ui.randomization.checked ? 0.9 + Math.random() * 0.2 : 1;
    if (this.groundGeomId >= 0) {
      const friction = ui.randomization.checked ? 0.55 + Math.random() * 0.75 : 1;
      this.model.geom_friction[this.groundGeomId * 3] = friction;
      this.model.geom_friction[this.groundGeomId * 3 + 1] = 0.005;
      this.model.geom_friction[this.groundGeomId * 3 + 2] = 0.0001;
    }
    this.nextRandomPush = 5 + Math.random() * 5;
    this.mujoco.mj_forward(this.model, this.data);
    this.history.reset(this.observe());
    this.lastTimestamp = performance.now();
  }

  private observe(): Record<string, number[]> {
    const quaternion = [this.data.qpos[3], this.data.qpos[4], this.data.qpos[5], this.data.qpos[6]];
    const [w, x, y, z] = quaternion as [number, number, number, number];
    const projectedGravity = [
      -2 * (x * z - w * y),
      -2 * (y * z + w * x),
      -(1 - 2 * (x * x + y * y)),
    ];
    const angularVelocity = [this.data.sensordata[4], this.data.sensordata[5], this.data.sensordata[6]];
    const jointPosition: number[] = [];
    const jointVelocity: number[] = [];
    for (let index = 0; index < this.manifest.output_size; index++) {
      jointPosition.push(this.data.qpos[7 + index] - this.manifest.action_center[index]);
      jointVelocity.push(this.data.qvel[6 + index] * 0.05);
    }
    return {
      base_ang_vel: angularVelocity.map((value) => value * 0.2),
      projected_gravity: projectedGravity,
      joint_pos_rel: jointPosition,
      joint_vel_rel: jointVelocity,
      last_action: Array.from(this.action),
      height_command: [this.targetHeight],
    };
  }

  private async runPolicy(): Promise<void> {
    if (this.policyPending) return;
    this.policyPending = true;
    try {
      this.history.push(this.observe());
      const tensor = new ort.Tensor("float32", this.history.flatten(), [1, this.manifest.input_size]);
      const result = await this.session.run({[this.manifest.input_name]: tensor});
      const output = result[this.manifest.output_name];
      if (!output) throw new Error(`Missing ONNX output ${this.manifest.output_name}`);
      const values = output.data as Float32Array;
      for (let index = 0; index < this.action.length; index++) {
        this.action[index] = THREE.MathUtils.clamp(
          values[index] ?? 0,
          this.manifest.action_clip[0],
          this.manifest.action_clip[1],
        );
      }
    } catch (error) {
      console.error(error);
      ui.statusDot.classList.add("paused");
      ui.statusText.textContent = "策略异常";
    } finally {
      this.policyPending = false;
    }
  }

  private applyPush(x: number, y: number): void {
    if (!this.data || this.trunkBodyId < 0) return;
    const scale = Number(ui.forceSlider.value);
    const length = Math.hypot(x, y) || 1;
    const force = 38 * scale;
    this.pushX = (force * x) / length;
    this.pushY = (force * y) / length;
    this.pushStepsRemaining = 24;
    this.pushMagnitude = force;
  }

  private updateExternalForce(): void {
    const offset = this.trunkBodyId * 6;
    for (let index = 0; index < 6; index++) this.data.xfrc_applied[offset + index] = 0;
    if (this.pushStepsRemaining > 0) {
      this.pushStepsRemaining--;
    } else {
      this.pushX = 0;
      this.pushY = 0;
      this.pushMagnitude = 0;
    }
    if (ui.randomization.checked && this.data.time >= this.nextRandomPush) {
      const angle = Math.random() * Math.PI * 2;
      this.applyPush(Math.cos(angle), Math.sin(angle));
      this.nextRandomPush = this.data.time + 5 + Math.random() * 5;
    }
    if (ui.stabilizer.checked) this.applyDemoStabilizer(offset);
    this.data.xfrc_applied[offset] += this.pushX;
    this.data.xfrc_applied[offset + 1] += this.pushY;
  }

  private applyDemoStabilizer(offset: number): void {
    const support = THREE.MathUtils.clamp((this.targetHeight + 0.1) / 0.4, 0, 1);
    if (support <= 0) return;
    const browserTrackedOffset = this.manifest.tracked_point_offset_m - 0.028;
    const targetRootHeight = this.targetHeight - browserTrackedOffset;
    const rootX = this.data.qpos[0];
    const rootY = this.data.qpos[1];
    const rootZ = this.data.qpos[2];
    const [w, x, y, z] = [this.data.qpos[3], this.data.qpos[4], this.data.qpos[5], this.data.qpos[6]];
    const bodyUpX = 2 * (x * z + w * y);
    const bodyUpY = 2 * (y * z - w * x);
    const mass = 19.666;
    const vertical = 760 * (targetRootHeight - rootZ) - 95 * this.data.qvel[2];
    this.data.xfrc_applied[offset] += support * (-46 * rootX - 13 * this.data.qvel[0]);
    this.data.xfrc_applied[offset + 1] += support * (-46 * rootY - 13 * this.data.qvel[1]);
    this.data.xfrc_applied[offset + 2] += support * THREE.MathUtils.clamp(vertical, -0.05 * mass * 9.81, 0.7 * mass * 9.81);
    this.data.xfrc_applied[offset + 3] += support * (90 * bodyUpY - 8 * this.data.qvel[3]);
    this.data.xfrc_applied[offset + 4] += support * (-90 * bodyUpX - 8 * this.data.qvel[4]);
    this.data.xfrc_applied[offset + 5] += support * (-3 * this.data.qvel[5]);
  }

  private assistedCenter(index: number): number {
    const name = this.manifest.joint_names[index];
    const crouch = THREE.MathUtils.clamp((0.72 - this.targetHeight) / 0.36, 0, 1);
    let offset = 0;
    if (name.endsWith("Hip_Pitch")) offset = -0.8 * crouch;
    if (name.endsWith("Knee_Pitch")) offset = 2.4 * crouch;
    if (name.endsWith("Ankle_Pitch")) offset = -1.2 * crouch;
    return THREE.MathUtils.clamp(
      this.manifest.action_center[index] + offset,
      this.manifest.position_minimum[index],
      this.manifest.position_maximum[index],
    );
  }

  private applyController(): void {
    this.data.qfrc_applied.fill(0);
    for (let index = 0; index < this.manifest.output_size; index++) {
      const assisted = ui.stabilizer.checked;
      const policyBlend = assisted ? 0.05 : 1;
      const desired =
        (assisted ? this.assistedCenter(index) : this.manifest.action_center[index]) +
        policyBlend * this.manifest.action_scale[index] * this.action[index];
      const positionError = desired - this.data.qpos[7 + index];
      const velocity = this.data.qvel[6 + index];
      const rawTorque =
        this.gainScale * this.manifest.stiffness[index] * positionError -
        this.gainScale * this.manifest.damping[index] * velocity;
      const limit = this.manifest.torque_limit[index];
      this.data.qfrc_applied[6 + index] = THREE.MathUtils.clamp(rawTorque, -limit, limit);
    }
  }

  private stepPhysics(): void {
    this.updateExternalForce();
    this.applyController();
    const integrationSubsteps = Math.max(
      1,
      Math.round((1 / this.manifest.simulation_rate_hz) / Number(this.model.opt.timestep)),
    );
    for (let index = 0; index < integrationSubsteps; index++) {
      this.mujoco.mj_step(this.model, this.data);
    }
    this.physicsSteps++;
    const policyDivider = Math.round(this.manifest.simulation_rate_hz / this.manifest.policy_rate_hz);
    if (this.physicsSteps % policyDivider === 0) void this.runPolicy();
  }

  private geometryFor(mjvGeom: any): THREE.BufferGeometry {
    const type = Number(mjvGeom.type);
    const dataId = Number(mjvGeom.dataid);
    const size = Array.from(mjvGeom.size, Number) as number[];
    const key = JSON.stringify([type, dataId, size]);
    const cached = this.geometryCache.get(key);
    if (cached) return cached;

    let geometry: THREE.BufferGeometry;
    const geom = this.mujoco.mjtGeom;
    if (type === geom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(size[0], 24, 16);
    } else if (type === geom.mjGEOM_CAPSULE.value) {
      geometry = new CapsuleGeometry(size[0], 2 * size[2]);
    } else if (type === geom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    } else if (type === geom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], 2 * size[2], 24);
      geometry.rotateX(Math.PI / 2);
    } else if (type === geom.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1, 24, 16);
      geometry.scale(size[0], size[1], size[2]);
    } else if (type === geom.mjGEOM_MESH.value && dataId >= 0) {
      // mjvGeom encodes mesh data as 2 * mesh_id, plus one for convex hulls.
      const meshId = Math.floor(dataId / 2);
      const vertexAddress = this.model.mesh_vertadr[meshId];
      const vertexCount = this.model.mesh_vertnum[meshId];
      const faceAddress = this.model.mesh_faceadr[meshId];
      const faceCount = this.model.mesh_facenum[meshId];
      const positions = new Float32Array(vertexCount * 3);
      const indices = new Uint32Array(faceCount * 3);
      for (let index = 0; index < positions.length; index++) {
        positions[index] = this.model.mesh_vert[vertexAddress * 3 + index];
      }
      for (let index = 0; index < indices.length; index++) {
        indices[index] = this.model.mesh_face[faceAddress * 3 + index];
      }
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setIndex(new THREE.BufferAttribute(indices, 1));
      geometry.computeVertexNormals();
    } else {
      geometry = new THREE.BufferGeometry();
    }
    this.geometryCache.set(key, geometry);
    return geometry;
  }

  private updateRobotScene(): void {
    this.mujoco.mjv_updateScene(
      this.model,
      this.data,
      this.mjvOption,
      this.mjvPerturb,
      this.mjvCamera,
      this.mujoco.mjtCatBit.mjCAT_ALL.value,
      this.mjvScene,
    );
    const geoms = this.mjvScene.geoms;
    const count = Number(geoms.size());
    for (let index = 0; index < count; index++) {
      const mjvGeom = geoms.get(index);
      if (!mjvGeom) continue;
      const rgba = Array.from(mjvGeom.rgba, Number) as number[];
      const mat = Array.from(mjvGeom.mat, Number) as number[];
      const pos = Array.from(mjvGeom.pos, Number) as number[];
      let mesh = this.meshes[index];
      if (!mesh) {
        const material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          roughness: 0.58,
          metalness: 0.18,
          transparent: rgba[3] < 0.999,
          opacity: rgba[3],
        });
        mesh = new THREE.Mesh(this.geometryFor(mjvGeom), material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.matrixAutoUpdate = false;
        this.meshes.push(mesh);
        this.scene.add(mesh);
      }
      mesh.visible = rgba[3] > 0.01;
      const material = mesh.material as THREE.MeshStandardMaterial;
      material.color.setRGB(rgba[0], rgba[1], rgba[2]);
      material.opacity = rgba[3];
      mesh.matrix.set(
        mat[0], mat[3], mat[6], pos[0],
        mat[1], mat[4], mat[7], pos[1],
        mat[2], mat[5], mat[8], pos[2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
      mjvGeom.delete();
    }
    for (let index = count; index < this.meshes.length; index++) this.meshes[index].visible = false;
    geoms.delete();
  }

  private updateTelemetry(): void {
    const actual = this.data.qpos[2] + this.manifest.tracked_point_offset_m - 0.028;
    ui.actualHeight.textContent = actual.toFixed(3);
    ui.heightError.textContent = Math.abs(this.targetHeight - actual).toFixed(3);
    ui.pushForce.textContent = this.pushMagnitude.toFixed(0);
  }

  private animate(timestamp: number): void {
    const frameTime = Math.min((timestamp - this.lastTimestamp) / 1000, 0.05);
    this.lastTimestamp = timestamp;
    if (!this.paused) {
      this.accumulator += frameTime;
      const timestep = 1 / this.manifest.simulation_rate_hz;
      let substeps = 0;
      while (this.accumulator >= timestep && substeps < 12) {
        this.stepPhysics();
        this.accumulator -= timestep;
        substeps++;
      }
    }
    this.updateRobotScene();
    this.updateTelemetry();
    this.controls.target.lerp(new THREE.Vector3(this.data.qpos[0], this.data.qpos[1], 0.44), 0.03);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame((time) => this.animate(time));
  }
}

const demo = new K1HeightDemo();
demo.initialize().catch((error) => {
  console.error(error);
  ui.loadingStage.textContent = error instanceof Error ? error.message : "仿真载入失败";
  ui.loadingProgress.style.background = "#ff7d62";
  ui.statusDot.classList.add("paused");
  ui.statusText.textContent = "载入失败";
});
