import fs from "fs";
import path from "path";
import inference from "../inference/inference";

const files = [
  "D:/VSC FILES/testtrain/food_not_food_ds/not_food/test/000005_IMG_20251028_171218.jpg",
  "D:/VSC FILES/testtrain/food_not_food_ds/not_food/test/000011_non_food_000626.jpg",
  "D:/VSC FILES/testtrain/food_not_food_ds/not_food/test/000013_126_bec2d20aab7396f549eacc56f25e9cd6.jpg.jpg",
  "D:/VSC FILES/testtrain/food_not_food_ds/not_food/test/000003_296906e71c7fbc16de927c9a8c8d35c1.jpg",
  "D:/VSC FILES/testtrain/food_not_food_ds/food/test/eggs_benedict_test_eggs_benedict_169.jpg",
  "D:/VSC FILES/testtrain/food_not_food_ds/food/test/donuts_test_donuts_41.jpg",
  "D:/VSC FILES/testtrain/food_not_food_ds/food/test/sipo_egg_test_sipo_egg_274.jpg",
  "D:/VSC FILES/testtrain/food_not_food_ds/food/test/strawberry_shortcake_test_strawberry_shortcake_190.jpg",
];

async function run() {
  try {
    console.log("Warming up model...");
    await inference.warmup();

    for (const f of files) {
      const abs = path.resolve(f);
      if (!fs.existsSync(abs)) {
        console.warn("File not found:", abs);
        continue;
      }
      const buf = fs.readFileSync(abs);
      try {
        const { label, scores, probs, topIndex } =
          (await inference.predictFromBuffer(buf)) as any;
        const out = label === "food" ? "fgod" : "not_food";
        const probsStr = probs
          ? probs.map((p: number) => p.toFixed(4)).join(",")
          : "";
        const topProb = probs ? probs[topIndex] : null;
        console.log(
          `${path.basename(abs)} -> model:${label} -> result:${out} top_prob:${
            topProb?.toFixed(4) ?? "n/a"
          } probs:[${probsStr}] logits:[${scores
            .map((s: number) => s.toFixed(4))
            .join(",")}]`
        );
      } catch (err: any) {
        console.error("Error predicting", abs, err?.message ?? err);
      }
    }
  } catch (err: any) {
    console.error("Fatal error in tests:", err?.message ?? err);
  }
}

run();
