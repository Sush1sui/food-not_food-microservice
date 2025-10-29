// Simple ping helper that calls SERVER_URL from environment every 10-14 minutes (randomized).
// Usage: import { startPingLoop, stopPingLoop } from './utils/pinger'; startPingLoop();

const MIN_MS = 10 * 60 * 1000; // 10 minutes
const MAX_MS = 14 * 60 * 1000; // 14 minutes

let timer: NodeJS.Timeout | null = null;

function randomIntervalMs() {
  return MIN_MS + Math.floor(Math.random() * (MAX_MS - MIN_MS + 1));
}

export async function pingOnce(): Promise<void> {
  const url = process.env.SERVER_URL;
  if (!url) {
    console.warn("pingOnce: SERVER_URL not set in env");
    return;
  }

  try {
    const headers: Record<string, string> = {};
    if (process.env.API_KEY) headers["x-api-key"] = process.env.API_KEY;

    // Use global fetch if available (Node 18+). This avoids extra deps.
    const fetchFn = (globalThis as any).fetch;
    if (typeof fetchFn !== "function") {
      // dynamic import of node-fetch as a fallback
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const nf = require("node-fetch");
      await nf(url, { headers });
      console.log("Pinged", url);
      return;
    }

    const resp = await fetchFn(url, { headers });
    console.log("Pinged", url, "status", resp.status);
  } catch (err) {
    console.warn("Ping failed for", process.env.SERVER_URL, err);
  }
}

function scheduleNext() {
  const ms = randomIntervalMs();
  timer = setTimeout(async () => {
    try {
      await pingOnce();
    } catch (e) {
      console.warn("pingOnce error", e);
    }
    scheduleNext();
  }, ms);
}

export function startPingLoop() {
  if (timer) return;
  if (!process.env.SERVER_URL)
    throw new Error("SERVER_URL not set in environment");
  scheduleNext();
}

export function stopPingLoop() {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

export default { pingOnce, startPingLoop, stopPingLoop };
