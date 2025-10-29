import { Request, Response, NextFunction } from "express";

// Middleware to check API key for server-to-server calls.
export function requireApiKey(req: Request, res: Response, next: NextFunction) {
  const expected = process.env.API_KEY;
  if (!expected) {
    console.warn("API_KEY not configured in environment");
    return res.status(500).json({ error: "Server misconfiguration" });
  }

  // Check header x-api-key, Authorization: ApiKey <key> or Bearer <key>, or query param api_key
  const headerKey = (
    req.header("x-api-key") ||
    req.header("X-API-KEY") ||
    ""
  ).toString();
  const auth = (
    req.header("authorization") ||
    req.header("Authorization") ||
    ""
  ).toString();
  const queryKey = (req.query?.api_key as string) || "";

  let provided = "";
  if (headerKey) provided = headerKey;
  else if (auth) {
    // Accept 'ApiKey <key>' or 'Bearer <key>' or raw key
    const parts = auth.split(" ");
    provided = parts.length > 1 ? parts[1] : parts[0];
  } else if (queryKey) provided = queryKey;

  if (!provided) {
    return res.status(401).json({ error: "Missing API key" });
  }

  if (provided !== expected) {
    return res.status(403).json({ error: "Invalid API key" });
  }

  return next();
}

export default requireApiKey;
