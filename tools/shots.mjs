// Look at a chapter in a real browser.
//
// Most of this project's bugs have been visual, and none of them were visible to the test
// suite: AppTest reports element values, not rendered geometry. It cannot see a diagram
// that failed to parse, a dark shape on a dark page, or a clipped figure.
//
//   npm install playwright && npx playwright install chromium
//   uv run streamlit run app/Home.py --server.port 8555 --server.headless true &
//   node tools/shots.mjs 12_arrows_and_grids 0,1,5 grid
//
// Step indexes are 0-based; the shots land in /tmp/<prefix>_<step>.png.
import { chromium } from "playwright";

const [slug, stepsArg, prefix] = process.argv.slice(2);
if (!slug || !stepsArg || !prefix) {
  console.error("usage: node tools/shots.mjs <slug> <step,step,...> <prefix>");
  process.exit(1);
}
const wanted = stepsArg.split(",").map((s) => parseInt(s, 10)).sort((a, b) => a - b);
const width = parseInt(process.env.KIDSML_WIDTH || "1280", 10);
const port = process.env.KIDSML_PORT || "8555";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width, height: 1200 } });
page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
await page.goto(`http://localhost:${port}/${slug}`, { waitUntil: "networkidle" });
await page.waitForTimeout(4000);

let at = 0;
for (const target of wanted) {
  while (at < target) {
    await page
      .locator('[data-testid="stButton"] button')
      .filter({ hasText: /Next/ })
      .last()
      .click({ timeout: 25000 });
    await page.waitForTimeout(1600);
    at++;
  }
  await page.waitForTimeout(2200);
  const title = await page.locator(".kml-step-title h2").first().textContent().catch(() => "?");
  const file = `/tmp/${prefix}_${target}.png`;
  await page.screenshot({ path: file, fullPage: true });
  console.log(`${target}: ${title} -> ${file}`);
}

const [scrollWidth, clientWidth] = await page.evaluate(() => [
  document.documentElement.scrollWidth,
  document.documentElement.clientWidth,
]);
if (scrollWidth > clientWidth) {
  console.log(`WARNING: horizontal scroll — ${scrollWidth}px of content in ${clientWidth}px`);
}
await browser.close();
