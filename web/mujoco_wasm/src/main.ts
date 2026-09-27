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
  position_delta_clip: [number, number];
  position_target_margin: number;
  position_braking_horizon_s: number;
  actuator_delay_steps: number[];
  position_target_velocity_limit: number[];
  height_residual_fade_joint_indices: number[];
  height_residual_fade_range: [number, number];
  height_residual_minimum_scale: number;
  height_posture_center: number[];
  height_posture_range: [number, number];
  height_posture_exponent: number;
  height_posture_blend: number;
  action_center: number[];
  action_scale: number[];
  position_minimum: number[];
  position_maximum: number[];
  stiffness: number[];
  damping: number[];
  torque_limit: number[];
  motor_effort_limit: number[];
  motor_velocity_limit: number[];
  motor_knee_velocity: number[];
  joint_armature: number[];
  policy_rate_hz: number;
  simulation_rate_hz: number;
  tracked_point_offset_m: number;
  standing_root_height_m: number;
};

type ModelAsset = {path: string; bytes: Uint8Array};
type ModelBundleHeader = {version: number; files: Array<{path: string; size: number}>};

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
  private session!: ort.InferenceSession;
  private manifest!: PolicyManifest;
  private history!: HistoryBank;

  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(42, 1, 0.02, 50);
  private readonly renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
  private readonly controls = new OrbitControls(this.camera, this.renderer.domElement);
  private readonly renderGeoms: Array<{geomId: number; mesh: THREE.Mesh}> = [];
  private readonly geometryCache = new Map<string, THREE.BufferGeometry>();

  private action = new Float32Array(22);
  private jointTargets = new Float32Array(22);
  private targetDelayHistory: Float32Array[] = [];
  private targetHeight = 0.72;
  private commandHeight = 0.72;
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
  private loadingProgress = 0;

  async initialize(): Promise<void> {
    this.setLoading("并行载入仿真资源", 8);
    const mujocoPromise = this.runtimeWasmSource("mujoco.wasm")
      .then((source) =>
        loadMujoco(typeof source === "string" ? {locateFile: () => source} : {wasmBinary: source}).then((runtime) => {
          this.setLoading("MuJoCo 已就绪", 36);
          return runtime;
        }),
      )
      .catch((error) => {
        throw new Error(`MuJoCo 载入失败: ${error instanceof Error ? error.message : String(error)}`, {cause: error});
      });
    const modelAssetsPromise = this.fetchModelAssets()
      .then((assets) => {
        this.setLoading("K1 模型已就绪", 52);
        return assets;
      })
      .catch((error) => {
        throw new Error(`K1 模型载入失败: ${error instanceof Error ? error.message : String(error)}`, {cause: error});
      });
    const manifestPromise = this.fetchJson<PolicyManifest>(`${BASE_URL}policy/k1_height_policy.json`).catch((error) => {
      throw new Error(`策略配置载入失败: ${error instanceof Error ? error.message : String(error)}`, {cause: error});
    });
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    const sessionPromise = this.runtimeWasmSource("ort-wasm-simd-threaded.wasm")
      .then((source) => {
        if (typeof source === "string") {
          ort.env.wasm.wasmPaths = {wasm: source};
        } else {
          ort.env.wasm.wasmBinary = source;
        }
        return ort.InferenceSession.create(`${BASE_URL}policy/k1_height_policy.onnx`, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        }).then((session) => {
          this.setLoading("ONNX 策略已就绪", 72);
          return session;
        });
      })
      .catch((error) => {
        throw new Error(`ONNX 策略载入失败: ${error instanceof Error ? error.message : String(error)}`, {cause: error});
      });

    const [mujoco, modelAssets, manifest, session] = await Promise.all([
      mujocoPromise,
      modelAssetsPromise,
      manifestPromise,
      sessionPromise,
    ]);
    this.mujoco = mujoco;
    this.manifest = manifest;
    this.session = session;
    this.installModelAssets(modelAssets);
    this.model = this.mujoco.MjModel.mj_loadXML("/working/K1_22dof.xml");
    for (let index = 0; index < this.manifest.output_size; index++) {
      this.model.dof_armature[6 + index] = this.manifest.joint_armature[index];
    }
    this.data = new this.mujoco.MjData(this.model);
    this.trunkBodyId = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_BODY.value, "Trunk");
    this.groundGeomId = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_GEOM.value, "ground");

    this.history = new HistoryBank(this.manifest);

    this.setLoading("初始化控制器", 88);
    this.setupRenderer();
    this.bindControls();
    this.reset();
    this.initializeRobotScene();
    await this.runPolicy();

    this.setLoading("就绪", 100);
    ui.statusDot.classList.remove("loading");
    ui.statusText.textContent = "运行中";
    window.setTimeout(() => ui.loading.classList.add("hidden"), 220);
    requestAnimationFrame((time) => this.animate(time));
  }

  private setLoading(label: string, progress: number): void {
    ui.loadingStage.textContent = label;
    this.loadingProgress = Math.max(this.loadingProgress, progress);
    ui.loadingProgress.style.width = `${this.loadingProgress}%`;
  }

  private async fetchBytes(url: string, attempts = 3): Promise<Uint8Array> {
    let lastError: unknown;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 60_000);
      try {
        const response = await fetch(url, {signal: controller.signal});
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return new Uint8Array(await response.arrayBuffer());
      } catch (error) {
        lastError = error;
        if (attempt < attempts) await new Promise((resolve) => window.setTimeout(resolve, 600 * attempt));
      } finally {
        window.clearTimeout(timeout);
      }
    }
    throw new Error(`无法下载 ${url}: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
  }

  private async fetchJson<T>(url: string): Promise<T> {
    const bytes = await this.fetchBytes(url);
    return JSON.parse(new TextDecoder().decode(bytes)) as T;
  }

  private arrayBuffer(bytes: Uint8Array): ArrayBuffer {
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    return copy.buffer;
  }

  private async gunzip(bytes: Uint8Array): Promise<Uint8Array> {
    const stream = new Blob([this.arrayBuffer(bytes)]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  private async runtimeWasmSource(name: string): Promise<string | Uint8Array> {
    const fallbackUrl = name === "mujoco.wasm" ? mujocoWasmUrl : ortWasmUrl;
    if (typeof (globalThis as any).DecompressionStream !== "function") {
      const url = new URL(fallbackUrl, location.href).href;
      ui.loadingStage.dataset.lastUrl = url;
      return url;
    }
    const compressedUrl = new URL(`${BASE_URL}vendor/${name}.gz.bin`, window.location.href).href;
    ui.loadingStage.dataset.lastUrl = compressedUrl;
    const compressed = await this.fetchBytes(compressedUrl);
    return this.gunzip(compressed);
  }

  private async fetchModelAssets(): Promise<ModelAsset[]> {
    if (typeof (globalThis as any).DecompressionStream !== "function") {
      const files = await this.fetchJson<string[]>(`${BASE_URL}models/K1/files.json`);
      return Promise.all(
        files.map(async (path) => ({path, bytes: await this.fetchBytes(`${BASE_URL}models/K1/${path}`)})),
      );
    }
    const compressed = await this.fetchBytes(`${BASE_URL}models/K1/assets.bin.gz.bin`);
    const bundle = await this.gunzip(compressed);
    const view = new DataView(bundle.buffer, bundle.byteOffset, bundle.byteLength);
    const headerLength = view.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(bundle.subarray(4, 4 + headerLength))) as ModelBundleHeader;
    if (header.version !== 1) throw new Error(`不支持的 K1 模型包版本: ${header.version}`);
    let offset = 4 + headerLength;
    return header.files.map((file) => {
      const bytes = bundle.slice(offset, offset + file.size);
      offset += file.size;
      return {path: file.path, bytes};
    });
  }

  private installModelAssets(files: ModelAsset[]): void {
    this.mujoco.FS.mkdir("/working");
    this.mujoco.FS.mkdir("/working/meshes");
    for (const file of files) {
      this.mujoco.FS.writeFile(`/working/${file.path}`, file.bytes);
    }
  }

  private setupRenderer(): void {
    this.scene.background = new THREE.Color(0xf7f8f7);
    this.scene.fog = new THREE.Fog(0xf7f8f7, 3.2, 8);

    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    ui.viewport.appendChild(this.renderer.domElement);

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1.35, -1.55, 0.94);
    this.controls.target.set(0, 0, 0.46);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 0.65;
    this.controls.maxDistance = 4;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    const hemi = new THREE.HemisphereLight(0xffffff, 0xcfd2d0, 2.15);
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
    const fill = new THREE.DirectionalLight(0xffffff, 1.25);
    fill.position.set(2.4, -1.4, 1.8);
    this.scene.add(fill);

    const groundMaterial = new THREE.MeshStandardMaterial({color: 0xebecea, roughness: 0.96, metalness: 0});
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(16, 16), groundMaterial);
    ground.receiveShadow = true;
    ground.position.z = -0.003;
    this.scene.add(ground);
    const grid = new THREE.GridHelper(8, 32, 0xc9ccca, 0xdfe1df);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.001;
    const materials = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const material of materials) {
      material.opacity = 0.55;
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
    this.targetHeight = THREE.MathUtils.clamp(value, 0.54, 0.72);
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
    ui.statusText.textContent = this.paused ? "已暂停" : "运行中";
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
      this.jointTargets[index] = this.manifest.action_center[index];
    }
    this.targetDelayHistory = [this.jointTargets.slice()];
    this.action.fill(0);
    this.pushStepsRemaining = 0;
    this.pushMagnitude = 0;
    this.pushX = 0;
    this.pushY = 0;
    this.physicsSteps = 0;
    this.accumulator = 0;
    this.commandHeight = this.targetHeight;
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
      height_command: [this.commandHeight],
    };
  }

  private async runPolicy(): Promise<void> {
    if (this.policyPending) return;
    this.policyPending = true;
    try {
      this.history.push(this.observe());
      const policyInput = this.history.flatten();
      const tensor = new ort.Tensor("float32", policyInput, [1, this.manifest.input_size]);
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
      this.updateJointTargets();
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
    const support = THREE.MathUtils.clamp((this.commandHeight + 0.1) / 0.4, 0, 1);
    if (support <= 0) return;
    const browserTrackedOffset = this.manifest.tracked_point_offset_m - 0.028;
    const targetRootHeight = this.commandHeight - browserTrackedOffset;
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
    const crouch = THREE.MathUtils.clamp((0.72 - this.commandHeight) / 0.36, 0, 1);
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

  private policyCenter(index: number): number {
    const [minimumHeight, maximumHeight] = this.manifest.height_posture_range;
    const phase = Math.pow(
      THREE.MathUtils.clamp(
        (maximumHeight - this.commandHeight) / (maximumHeight - minimumHeight),
        0,
        1,
      ),
      this.manifest.height_posture_exponent,
    );
    return (
      this.manifest.action_center[index] +
      this.manifest.height_posture_blend *
        phase *
        (this.manifest.height_posture_center[index] - this.manifest.action_center[index])
    );
  }

  private updateJointTargets(): void {
    const [fadeStart, fadeEnd] = this.manifest.height_residual_fade_range;
    const phase = THREE.MathUtils.clamp((fadeEnd - this.commandHeight) / (fadeEnd - fadeStart), 0, 1);
    const smoothPhase = phase * phase * (3 - 2 * phase);
    const heightResidualScale =
      this.manifest.height_residual_minimum_scale +
      (1 - this.manifest.height_residual_minimum_scale) * smoothPhase;
    const fadedJoints = new Set(this.manifest.height_residual_fade_joint_indices);
    const assisted = ui.stabilizer.checked;
    const policyBlend = assisted ? 0.05 : 1;

    for (let index = 0; index < this.manifest.output_size; index++) {
      const jointDelta = THREE.MathUtils.clamp(
        this.manifest.action_scale[index] * this.action[index],
        this.manifest.position_delta_clip[0],
        this.manifest.position_delta_clip[1],
      );
      const residualScale = fadedJoints.has(index) ? heightResidualScale : 1;
      const center = assisted ? this.assistedCenter(index) : this.policyCenter(index);
      let requested = THREE.MathUtils.clamp(
        center + policyBlend * residualScale * jointDelta,
        this.manifest.position_minimum[index],
        this.manifest.position_maximum[index],
      );
      const safeMinimum = this.manifest.position_minimum[index] + this.manifest.position_target_margin;
      const safeMaximum = this.manifest.position_maximum[index] - this.manifest.position_target_margin;
      requested = THREE.MathUtils.clamp(requested, safeMinimum, safeMaximum);
      const current = this.data.qpos[7 + index];
      const velocity = this.data.qvel[6 + index];
      const predicted = current + velocity * this.manifest.position_braking_horizon_s;
      if ((velocity > 0 && predicted > safeMaximum) || (velocity < 0 && predicted < safeMinimum)) {
        requested = THREE.MathUtils.clamp(
          current - velocity * this.manifest.position_braking_horizon_s,
          safeMinimum,
          safeMaximum,
        );
      }
      const rateLimited =
        this.jointTargets[index] +
        THREE.MathUtils.clamp(
          requested - this.jointTargets[index],
          -this.manifest.position_target_velocity_limit[index] / this.manifest.policy_rate_hz,
          this.manifest.position_target_velocity_limit[index] / this.manifest.policy_rate_hz,
        );
      const errorLimit = this.manifest.torque_limit[index] / this.manifest.stiffness[index];
      this.jointTargets[index] = current + THREE.MathUtils.clamp(rateLimited - current, -errorLimit, errorLimit);
    }
  }

  private applyController(): void {
    this.data.qfrc_applied.fill(0);
    for (let index = 0; index < this.manifest.output_size; index++) {
      const current = this.data.qpos[7 + index];
      const delay = Math.max(0, Math.round(this.manifest.actuator_delay_steps[index]));
      const historyIndex = Math.max(0, this.targetDelayHistory.length - 1 - delay);
      const positionError = this.targetDelayHistory[historyIndex][index] - current;
      const velocity = this.data.qvel[6 + index];
      const rawTorque =
        this.gainScale * this.manifest.stiffness[index] * positionError -
        this.gainScale * this.manifest.damping[index] * velocity;
      const speed = Math.abs(velocity);
      const motorLimit = this.manifest.motor_effort_limit[index];
      const maximumVelocity = this.manifest.motor_velocity_limit[index];
      const kneeVelocity = Math.min(this.manifest.motor_knee_velocity[index], maximumVelocity);
      const speedLimit =
        speed <= kneeVelocity || kneeVelocity >= maximumVelocity
          ? motorLimit
          : THREE.MathUtils.clamp(
              (motorLimit * (maximumVelocity - speed)) / (maximumVelocity - kneeVelocity),
              0,
              motorLimit,
            );
      this.data.qfrc_applied[6 + index] = THREE.MathUtils.clamp(rawTorque, -speedLimit, speedLimit);
    }
  }

  private stepPhysics(): void {
    const maximumCommandStep = 0.08 / this.manifest.simulation_rate_hz;
    this.commandHeight += THREE.MathUtils.clamp(
      this.targetHeight - this.commandHeight,
      -maximumCommandStep,
      maximumCommandStep,
    );
    this.updateExternalForce();
    this.targetDelayHistory.push(this.jointTargets.slice());
    const maximumDelay = Math.max(...this.manifest.actuator_delay_steps);
    while (this.targetDelayHistory.length > maximumDelay + 1) this.targetDelayHistory.shift();
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

  private geometryFor(type: number, dataId: number, size: number[]): THREE.BufferGeometry {
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
      const vertexAddress = this.model.mesh_vertadr[dataId];
      const vertexCount = this.model.mesh_vertnum[dataId];
      const faceAddress = this.model.mesh_faceadr[dataId];
      const faceCount = this.model.mesh_facenum[dataId];
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

  private robotAppearance(meshName: string): THREE.MeshStandardMaterialParameters {
    if (meshName === "Trunk") {
      return {color: 0xd9dcdb, roughness: 0.3, metalness: 0.58};
    }
    if (meshName === "K1logo") {
      return {color: 0x303332, roughness: 0.42, metalness: 0.18};
    }
    if (meshName.endsWith("Ankle_Cross")) {
      return {color: 0xe05a1f, roughness: 0.36, metalness: 0.28};
    }
    return {color: 0x1b1e1f, roughness: 0.36, metalness: 0.24};
  }

  private initializeRobotScene(): void {
    for (let geomId = 0; geomId < Number(this.model.ngeom); geomId++) {
      if (Number(this.model.geom_group[geomId]) !== 1) continue;
      const rgbaOffset = geomId * 4;
      const rgba = [
        Number(this.model.geom_rgba[rgbaOffset]),
        Number(this.model.geom_rgba[rgbaOffset + 1]),
        Number(this.model.geom_rgba[rgbaOffset + 2]),
        Number(this.model.geom_rgba[rgbaOffset + 3]),
      ];
      const sizeOffset = geomId * 3;
      const size = [
        Number(this.model.geom_size[sizeOffset]),
        Number(this.model.geom_size[sizeOffset + 1]),
        Number(this.model.geom_size[sizeOffset + 2]),
      ];
      const type = Number(this.model.geom_type[geomId]);
      const dataId = Number(this.model.geom_dataid[geomId]);
      const meshName =
        type === this.mujoco.mjtGeom.mjGEOM_MESH.value && dataId >= 0
          ? this.mujoco.mj_id2name(this.model, this.mujoco.mjtObj.mjOBJ_MESH.value, dataId) ?? ""
          : "";
      const material = new THREE.MeshStandardMaterial({
        ...this.robotAppearance(meshName),
        transparent: rgba[3] < 0.999,
        opacity: rgba[3],
      });
      const mesh = new THREE.Mesh(this.geometryFor(type, dataId, size), material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.visible = rgba[3] > 0.01;
      this.renderGeoms.push({geomId, mesh});
      this.scene.add(mesh);
    }
    this.updateRobotScene();
  }

  private updateRobotScene(): void {
    for (const {geomId, mesh} of this.renderGeoms) {
      const matOffset = geomId * 9;
      const posOffset = geomId * 3;
      const mat = this.data.geom_xmat;
      const pos = this.data.geom_xpos;
      mesh.matrix.set(
        mat[matOffset], mat[matOffset + 1], mat[matOffset + 2], pos[posOffset],
        mat[matOffset + 3], mat[matOffset + 4], mat[matOffset + 5], pos[posOffset + 1],
        mat[matOffset + 6], mat[matOffset + 7], mat[matOffset + 8], pos[posOffset + 2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;
    }
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
  const message = error instanceof Error ? error.message : "仿真载入失败";
  ui.loadingStage.textContent = message;
  ui.loadingStage.title = error instanceof Error ? error.stack ?? message : message;
  (window as any).__k1LoadError = error;
  ui.loadingProgress.style.background = "#ff7d62";
  ui.statusDot.classList.add("paused");
  ui.statusText.textContent = "载入失败";
});
