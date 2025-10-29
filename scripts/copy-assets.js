const fs = require("fs");
const path = require("path");

const projectRoot = path.resolve(__dirname, "..");
const srcDir = path.join(projectRoot, "src", "inference");
const outDir = path.join(projectRoot, "dist", "inference");

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function copyIfExists(name) {
  const src = path.join(srcDir, name);
  const dest = path.join(outDir, name);
  if (fs.existsSync(src)) {
    ensureDir(outDir);
    fs.copyFileSync(src, dest);
    console.log(`Copied ${name} -> dist/inference`);
    return true;
  }
  console.warn(`Asset not found in src/inference: ${name}`);
  return false;
}

const assets = ["model.onnx", "class_names.json"];
let copied = 0;
for (const a of assets) if (copyIfExists(a)) copied++;

if (!copied)
  console.warn(
    "No inference assets copied. Ensure model.onnx and/or class_names.json exist in src/inference."
  );
else console.log(`Copied ${copied} asset(s) to dist/inference`);

process.exit(0);
