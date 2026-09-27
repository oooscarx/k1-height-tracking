import {mkdir, readFile, rm, writeFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";
import {gzipSync} from "node:zlib";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(appDir, "../..");
const sourceDir = path.join(repoRoot, "third_party/booster_assets/robots/K1");
const outputDir = path.join(appDir, "public/models/K1");
const vendorDir = path.join(appDir, "public/vendor");
const legacyOrtDir = path.join(appDir, "public/ort");

await rm(outputDir, {recursive: true, force: true});
await rm(legacyOrtDir, {recursive: true, force: true});
await mkdir(path.join(outputDir, "meshes"), {recursive: true});

const sourceXml = await readFile(path.join(sourceDir, "K1_22dof.xml"), "utf8");
const browserXml = sourceXml.replace(
  'name="ground" type="plane" pos="0 0 0" size="0 0 1" material="matplane" condim="1" friction="0.4 0.005 0.0001"',
  'name="ground" type="plane" pos="0 0 0" size="0 0 1" material="matplane" condim="3" friction="1.0 0.005 0.0001"',
);
if (browserXml === sourceXml) throw new Error("K1 browser ground contact patch no longer matches the source MJCF");
await writeFile(path.join(outputDir, "K1_22dof.xml"), browserXml);

const meshFiles = [...new Set([...browserXml.matchAll(/file="([^"]+\.STL)"/g)].map((match) => match[1]))].sort();
const bundledFiles = [{path: "K1_22dof.xml", data: Buffer.from(browserXml)}];
for (const file of meshFiles) {
  const data = await readFile(path.join(sourceDir, "meshes", file));
  await writeFile(path.join(outputDir, "meshes", file), data);
  bundledFiles.push({path: `meshes/${file}`, data});
}
await writeFile(
  path.join(outputDir, "files.json"),
  JSON.stringify(["K1_22dof.xml", ...meshFiles.map((file) => `meshes/${file}`)], null, 2) + "\n",
);
const meshAssets = [...browserXml.matchAll(/<mesh\s+name="([^"]+)"\s+file="([^"]+\.STL)"/g)].map((match) => ({
  name: match[1],
  file: match[2],
}));
await writeFile(path.join(outputDir, "meshes.json"), JSON.stringify(meshAssets, null, 2) + "\n");

const bundleHeader = Buffer.from(
  JSON.stringify({version: 1, files: bundledFiles.map((file) => ({path: file.path, size: file.data.length}))}),
);
const bundleHeaderLength = Buffer.alloc(4);
bundleHeaderLength.writeUInt32LE(bundleHeader.length);
const modelBundle = gzipSync(
  Buffer.concat([bundleHeaderLength, bundleHeader, ...bundledFiles.map((file) => file.data)]),
  {level: 9},
);
await writeFile(path.join(outputDir, "assets.bin.gz.bin"), modelBundle);

await rm(vendorDir, {recursive: true, force: true});
await mkdir(vendorDir, {recursive: true});
const wasmAssets = [
  {
    source: path.join(appDir, "node_modules/@mujoco/mujoco/mujoco.wasm"),
    target: "mujoco.wasm",
  },
  {
    source: path.join(appDir, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"),
    target: "ort-wasm-simd-threaded.wasm",
  },
];
for (const asset of wasmAssets) {
  const data = await readFile(asset.source);
  await writeFile(path.join(vendorDir, `${asset.target}.gz.bin`), gzipSync(data, {level: 9}));
}

console.log(
  `Synced K1 XML and ${meshFiles.length} meshes (${(modelBundle.length / 1_000_000).toFixed(1)} MB compressed).`,
);
