import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import inference from "./inference/inference";
import { startPingLoop } from "./utils/pinger";
import { requireApiKey } from "./middleware/apiKey";

dotenv.config();

const upload = multer({ storage: multer.memoryStorage() });
const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;

// Root endpoint for health or external pings
app.get(
  "/",
  requireApiKey,
  async (_req: express.Request, res: express.Response) => {
    const ready = !!inference;
    const uptime = process.uptime();
    return res.json({ status: "ok", ready, uptime });
  }
);

// Keep /health for compatibility
app.get("/health", (_req: express.Request, res: express.Response) =>
  res.json({ status: "ok" })
);

// POST /predict, multipart form with field 'image'
app.post(
  "/predict",
  requireApiKey,
  upload.single("image"),
  async (req: express.Request, res: express.Response) => {
    try {
      if (!req.file || !req.file.buffer) {
        return res
          .status(400)
          .json({ error: "Missing image file (field name: image)" });
      }

      const { buffer } = req.file;

      const { label } = await inference.predictFromBuffer(buffer);

      // Map the model's class names ("food", "not_food") to the exact output.
      const out = label === "food" ? "food" : "not_food";

      return res.json({ result: out });
    } catch (err: any) {
      console.error("Prediction error:", err?.message ?? err);
      return res.status(500).json({ error: err?.message ?? String(err) });
    }
  }
);

// Warmup model on start
inference
  .warmup()
  .then(() => {
    console.log("Model warmed up");
  })
  .catch((e) => {
    console.warn(
      "Model warmup failed (will try on first request):",
      e?.message ?? e
    );
  });

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log('POST /predict with form-data field "image" (file)');
});

// Optionally start the ping loop when START_PINGER=true (reads SERVER_URL from .env)
if (process.env.START_PINGER === "true") {
  try {
    startPingLoop();
    console.log("Ping loop started (from env START_PINGER=true)");
  } catch (e) {
    console.warn("Failed to start ping loop:", e);
  }
}
