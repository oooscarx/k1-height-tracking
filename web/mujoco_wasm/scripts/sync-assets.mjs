import {cp, mkdir, readFile, rm, writeFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const appDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(appDir, "../..");
const sourceDir = path.join(repoRoot, "third_party/booster_assets/robots/K1");
const outputDir = path.join(appDir, "public/models/K1");

await rm(outputDir, {recursive: true, force: true});
await mkdir(path.join(outputDir, "meshes"), {recursive: true});

const sourceXml = await readFile(path.join(sourceDir, "K1_22dof.xml"), "utf8");
const browserXml = sourceXml;
await writeFile(path.join(outputDir, "K1_22dof.xml"), browserXml);

const meshFiles = [...new Set([...browserXml.matchAll(/file="([^"]+\.STL)"/g)].map((match) => match[1]))].sort();
for (const file of meshFiles) {
  await cp(path.join(sourceDir, "meshes", file), path.join(outputDir, "meshes", file));
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

console.log(`Synced K1 XML and ${meshFiles.length} meshes.`);
