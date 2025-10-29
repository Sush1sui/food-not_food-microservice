import path from "path";
import fs from "fs";
import sharp from "sharp";
import * as ort from "onnxruntime-node";

// Resolve assets from multiple possible locations so the service works
// whether running from source (ts-node) or from compiled `dist`.
function resolveAssetPath(filename: string) {
  const candidates = [
    path.resolve(__dirname, filename),
    path.resolve(__dirname, "..", "src", "inference", filename),
    path.resolve(process.cwd(), "src", "inference", filename),
    path.resolve(process.cwd(), "dist", "inference", filename),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch (e) {
      // ignore
    }
  }
  // fallback to runtime dir
  return path.resolve(__dirname, filename);
}

function getModelPath() {
  return resolveAssetPath("model.onnx");
}

function getClassNamesPath() {
  return resolveAssetPath("class_names.json");
}

let session: ort.InferenceSession | null = null;
let classNames: string[] = [];

async function loadClassNames() {
  if (classNames.length) return classNames;

  // 1) Try reading class_names.json if present
  try {
    const p = getClassNamesPath();
    if (fs.existsSync(p)) {
      const raw = await fs.promises.readFile(p, "utf8");
      classNames = JSON.parse(raw) as string[];
      return classNames;
    }
  } catch (e) {
    // ignore and try other strategies
  }

  // 2) Try to read labels from model metadata (if exporter embedded them)
  try {
    const sess = await loadSession();
    const meta =
      (sess as any).metadata ||
      (sess as any).modelMetadata ||
      (sess as any).customMetadataMap ||
      (sess as any).customMetadata;
    const map = meta && meta.customMetadataMap ? meta.customMetadataMap : meta;
    if (map && typeof map === "object") {
      const keys = [
        "labels",
        "classes",
        "class_names",
        "label_names",
        "labels_json",
      ];
      for (const k of keys) {
        const raw = map[k] || map[k.toLowerCase()] || map[k.toUpperCase()];
        if (raw && typeof raw === "string") {
          try {
            const parsed = JSON.parse(raw);
            if (
              Array.isArray(parsed) &&
              parsed.every((x) => typeof x === "string")
            ) {
              classNames = parsed;
              return classNames;
            }
          } catch (err) {
            const parts = raw
              .split(/[,;\n]+/)
              .map((s: string) => s.trim())
              .filter(Boolean);
            if (parts.length) {
              classNames = parts;
              return classNames;
            }
          }
        }
      }
    }
  } catch (e) {
    // ignore
  }

  // 3) Heuristic: scan ONNX bytes for JSON array or label text
  try {
    const buf = await fs.promises.readFile(getModelPath());
    const text = buf.toString("utf8");
    const arrMatch = text.match(/\[[^\]]+\]/);
    if (arrMatch) {
      try {
        const parsed = JSON.parse(arrMatch[0]);
        if (
          Array.isArray(parsed) &&
          parsed.every((x) => typeof x === "string")
        ) {
          classNames = parsed;
          return classNames;
        }
      } catch (e) {
        // ignore
      }
    }
    const keywordMatch = text.match(
      /(?:labels|class_names|classes)[^:\n\r]{0,120}([\w\s,\-\/]+)(?:\n|$)/i
    );
    if (keywordMatch && keywordMatch[1]) {
      const parts = keywordMatch[1]
        .split(/[,;\n]+/)
        .map((s: string) => s.trim())
        .filter(Boolean);
      if (parts.length) {
        classNames = parts;
        return classNames;
      }
    }
  } catch (e) {
    // ignore
  }

  // 4) Final fallback: reasonable defaults (don't crash)
  classNames = ["food", "not_food"];
  return classNames;
}

async function loadSession() {
  if (session) return session;
  // Create session (singleton)
  session = await ort.InferenceSession.create(getModelPath());
  return session;
}

// Default normalization/size if model metadata is not available.
// We'll try to read the expected input height/width from the model metadata
// and fall back to these values if unavailable.
// Model expects 252x252 in this project; use as fallback.
const DEFAULT_IMAGE_SIZE = 252;

// ImageNet-style normalization (RGB)
const NORMALIZE_MEAN = [0.485, 0.456, 0.406];
const NORMALIZE_STD = [0.229, 0.224, 0.225];

function toNumber(v: any): number | null {
  if (typeof v === "number") return v;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function preprocessToTensor(
  buffer: Buffer,
  targetWidth: number,
  targetHeight: number
) {
  return sharp(buffer)
    .resize(targetWidth, targetHeight)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
    .then(({ data, info }) => {
      const { width, height, channels } = info;
      // channels should be 3 (RGB)
      const floatData = new Float32Array(1 * 3 * height * width);
      // convert to NCHW and apply ImageNet normalization
      const hw = width * height;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * channels;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const px = y * width + x;
          // to [0,1]
          const rf = r / 255.0;
          const gf = g / 255.0;
          const bf = b / 255.0;
          // normalize per channel (mean/std), keep channel order RGB
          floatData[0 * hw + px] = (rf - NORMALIZE_MEAN[0]) / NORMALIZE_STD[0];
          floatData[1 * hw + px] = (gf - NORMALIZE_MEAN[1]) / NORMALIZE_STD[1];
          floatData[2 * hw + px] = (bf - NORMALIZE_MEAN[2]) / NORMALIZE_STD[2];
        }
      }
      // Return tensor data and shape (NCHW)
      return { data: floatData, shape: [1, 3, height, width] as number[] };
    });
}

function getModelExpectedHW(
  sess: ort.InferenceSession,
  inputName: string
): { height: number; width: number } {
  try {
    let meta = (sess as any).inputMetadata || (sess as any).inputs;
    let info: any;
    if (Array.isArray(meta)) {
      info = meta.find((m: any) => m.name === inputName) || meta[0];
    } else {
      info = meta ? meta[inputName] : undefined;
    }
    const dims = info
      ? info.shape ||
        info.dimensions ||
        info.dims ||
        info.type?.shape ||
        info.type?.dims
      : undefined;
    if (Array.isArray(dims)) {
      // Try to parse numbers from dims
      const parsed = dims.map((d: any) => toNumber(d));
      // If parsed contains 4 dims, handle NCHW or NHWC
      if (parsed.length === 4) {
        const [_d0, d1, d2, d3] = parsed;
        if (d1 === 3 && d2 && d3) {
          // NCHW
          return { height: d2, width: d3 } as any;
        }
        if (d3 === 3 && d1 && d2) {
          // NHWC
          return { height: d1, width: d2 } as any;
        }
      }
      // If 3 dims like [3,H,W]
      if (parsed.length === 3) {
        const [_d0, d1, d2] = parsed;
        if (_d0 === 3 && d1 && d2) return { height: d1, width: d2 } as any;
      }
      // As a fallback, find the last two numeric dims
      const nums = parsed.filter(
        (n: any) => typeof n === "number" && !Number.isNaN(n)
      );
      if (nums.length >= 2) {
        const h = nums[nums.length - 2];
        const w = nums[nums.length - 1];
        return { height: h, width: w } as any;
      }
    }
  } catch (e) {
    // ignore and fallback
  }
  // If we couldn't find dims, try to log the model input metadata for debugging
  try {
    const meta = (sess as any).inputMetadata || (sess as any).inputs;
    console.warn("Could not parse model input dims, inputMetadata:", meta);
  } catch {}
  return { height: DEFAULT_IMAGE_SIZE, width: DEFAULT_IMAGE_SIZE };
}

export async function predictFromBuffer(buffer: Buffer): Promise<{
  label: string;
  scores: number[];
  probs?: number[];
  topIndex: number;
}> {
  await loadClassNames();
  const sess = await loadSession();

  const inputMeta =
    (sess as any).inputNames && (sess as any).inputNames[0]
      ? (sess as any).inputNames[0]
      : Object.keys((sess as any).inputMetadata || {})[0];
  if (!inputMeta) throw new Error("Could not determine model input name");

  const { height: expectedH, width: expectedW } = getModelExpectedHW(
    sess,
    inputMeta
  );

  const { data, shape } = await preprocessToTensor(
    buffer,
    expectedW,
    expectedH
  );

  const tensor = new ort.Tensor("float32", data as Float32Array, shape);

  const feeds: Record<string, ort.Tensor> = {};
  feeds[inputMeta] = tensor;

  const results = await sess.run(feeds);
  // get first output
  const outVals = Object.values(results) as ort.Tensor[];
  if (!outVals.length) throw new Error("No outputs from model");

  const out = outVals[0];
  const logits = Array.from(out.data as Float32Array) as number[];

  // softmax to probabilities (numerically stable)
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sumExp);

  // argmax on probabilities
  let top = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[top]) top = i;
  }

  const label = classNames[top] ?? `${top}`;

  return { label, scores: logits, probs, topIndex: top };
}

export async function warmup() {
  try {
    await loadClassNames();
    await loadSession();
  } catch (err) {
    // swallow here; caller may handle
    throw err;
  }
}

export default { predictFromBuffer, warmup };
