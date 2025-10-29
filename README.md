# Food / Not-Food Inference Microservice

Lightweight Node.js (TypeScript) microservice that serves a binary food vs not_food ONNX model.

This repository exposes an HTTP API to run inference on images and is designed for server-to-server communication.

---

## Features

- POST `/predict` — send an image file (form-data) and receive `"food"` or `"not_food"` as the result.
- GET `/` — protected root endpoint returning basic status (used for internal pings).
- GET `/health` — lightweight public health check (no auth required).
- Optional background pinger to call your server URL periodically (10–14 minutes randomized), using the configured `API_KEY`.

---

## Quick start

1. Install dependencies:

```bash
npm install
```

2. Create a `.env` file in project root (do NOT commit this file):

```
API_KEY=your-secret-api-key
PORT=9669
SERVER_URL=http://localhost:9669/   # optional — used by pinger
START_PINGER=false                  # optional: set to 'true' to auto-start pinger
```

3. Start development server (ts-node):

```bash
npm run dev
```

4. Or build and run production bundle:

```bash
npm run build
npm start
```

---

## Endpoints

All protected endpoints require the API key. Provide it using the header `x-api-key` or `Authorization: ApiKey <key>`.

- POST /predict
  - Description: Run inference on an uploaded image.
  - Method: POST
  - Content-Type: multipart/form-data
  - Form field: `image` (file)
  - Auth: `x-api-key: <API_KEY>` header
  - Response: JSON { "result": "food" | "not_food" }

Example:

```bash
curl -H "x-api-key: $API_KEY" -F "image=@/path/to/photo.jpg" http://localhost:9669/predict
```

- GET /

  - Description: Protected root endpoint that returns { status, ready, uptime }
  - Use: suitable for internal pings (pinger uses this by default).

- GET /health
  - Description: Public health check (no API key required)
  - Response: { "status": "ok" }

---

## Authentication

This microservice uses a shared API key for server-to-server authentication. Set `API_KEY` in `.env`. The server accepts the key in:

- `x-api-key` header (recommended)
- `Authorization: ApiKey <key>` or `Authorization: Bearer <key>`
- `?api_key=` query param (less recommended)

Requests without a valid key receive `401` (missing) or `403` (invalid).

---

## Pinger (background helper)

The repository includes a pinger (`src/utils/pinger.ts`) that will call the configured `SERVER_URL` every 10–14 minutes at a randomized interval.

To enable it automatically, set in `.env`:

```
SERVER_URL=http://your-target-url/
START_PINGER=true
```

The pinger will include the `x-api-key` header (from `API_KEY`) when calling the URL so protected endpoints accept the ping.

You can also manually import and run the helper:

```ts
import { startPingLoop, stopPingLoop, pingOnce } from "./src/utils/pinger";
startPingLoop(); // start scheduled pings
pingOnce(); // single ping
stopPingLoop(); // stop
```

---

## Test runner (local images)

There's a small test runner at `src/test/run_tests.ts` that loads images from disk and runs `predictFromBuffer` directly. This is helpful to run batches of local images without calling the HTTP API.

Run it with:

```bash
npx ts-node src/test/run_tests.ts
```

The runner prints logits, softmax probabilities and the top-1 probability for each image.

---

## Model & preprocessing details

- Model: ONNX model located at `src/inference/model.onnx` (binary not ignored by default).
- Class mapping: `src/inference/class_names.json` — maps indices to labels (e.g. `["food","not_food"]`).
- Expected input:
  - Shape: `[1, 3, 252, 252]` (NCHW)
  - Channel order: RGB
  - Normalization: ImageNet-style mean/std per channel
    - mean = [0.485, 0.456, 0.406]
    - std = [0.229, 0.224, 0.225]
- Output: 1-D logits vector — code applies softmax to produce probabilities and uses argmax to choose the predicted class.

If your model uses different sizing/normalization, adjust `src/inference/inference.ts` accordingly.

---

## Development notes & troubleshooting

- Windows + `sharp`: installing `sharp` can require native dependencies. If you hit installation errors, consult the `sharp` docs (prebuilt binaries are preferred).
- `onnxruntime-node` sometimes requires native binaries; if you see errors loading the session, ensure the runtime supports your platform and Node version.
- If you see a model dimension mismatch error (e.g. expected 252 but got other size), confirm your `model.onnx` input metadata or edit the preprocessing target size in `src/inference/inference.ts`.

---

## Security

- Do not commit `.env` or secret values. `.gitignore` in this project already ignores `.env`.
- For higher security consider mutual TLS, signed requests, or rotating API keys.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes and run `npm run build` to ensure TypeScript compiles
4. Open a PR with a clear description

---
