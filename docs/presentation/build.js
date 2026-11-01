// Executive results deck for the BenchLab solar-wind forecasting entry (English, dark neon theme).
// The result figures come from make_figures.py (run it first); bar charts stay native PowerPoint charts.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const icons = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");
const D = require("./data.json");

const OUT = process.argv[2] || "benchlab_results.pptx";
const FIG = path.join(__dirname, "figures");

const THEME = {
  name: "Neon Corona",
  headFontFace: "Century Gothic",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "07060B", lt1: "F4F2FA", dk2: "16141F", lt2: "A7A3BA",
    accent1: "B026FF", accent2: "FFC72C", accent3: "39FF88",
    accent4: "7C6CF2", accent5: "5E5A70", accent6: "FF8A3D",
    hlink: "39FF88", folHlink: "B026FF",
  },
};
const H = THEME.colors; // hex, for options that only take hex
const CARD = "16141F";
const MUTED_LINE = "2C2938";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Forecasting the Solar Wind 72 Hours Ahead";
pres.author = "Adriano P. Almeida";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: H.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 7.6, h: 2.2, fontSize: 44, bold: true,
      color: C.background1, valign: "bottom", margin: 0, fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.45, w: 7.4, h: 1.2, fontSize: 18,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
// Action titles: one full-sentence message per slide under a small section kicker.
pres.defineSlideMaster({
  title: "CONTENT_DARK",
  background: { color: H.dk1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.62, w: 12.1, h: 0.62, fontSize: 26, bold: true,
      color: C.background1, valign: "middle", margin: 0, fontFace: THEME.headFontFace }, text: "" } },
    { text: { text: "FDL BenchLab · Solar Wind Prediction Tournament 2026",
      options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: H.accent5, align: "right", margin: 0 },
});

// ---------- helpers ----------
async function icon(name, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(icons[name], { color: "#" + color, size }));
  const png = await sharp(Buffer.from(svg)).resize(size, size).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}
// Motif: an icon inside a neon-outlined circle.
async function iconCircle(slide, name, color, x, y, d = 0.7, id = "icon") {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: CARD }, line: { color, width: 2 }, objectName: id + " ring" });
  slide.addImage({ data: await icon(name, color), x: x + d * 0.27, y: y + d * 0.27, w: d * 0.46, h: d * 0.46, objectName: id });
}
function card(slide, x, y, w, h, id, border = MUTED_LINE) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: CARD },
    line: { color: border, width: 1 }, objectName: id });
}
function text(slide, t, opts) {
  slide.addText(t, { isTextBox: true, margin: 0, color: C.background1, fontSize: 14, valign: "top", ...opts });
}
function chartBase(extra = {}) {
  return {
    catAxisLabelColor: H.lt2, valAxisLabelColor: H.lt2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLineShow: false, valAxisLineShow: false,
    valGridLine: { color: MUTED_LINE, size: 0.75 }, catGridLine: { style: "none" },
    legendColor: H.lt1, legendFontFace: "+mn-lt", legendFontSize: 11,
    valAxisTitleColor: H.lt2, catAxisTitleColor: H.lt2, valAxisTitleFontSize: 11, catAxisTitleFontSize: 11,
    valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
    dataLabelColor: H.lt1, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11,
    ...extra,
  };
}
function content(kicker, title, section) {
  const s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: section });
  text(s, kicker.toUpperCase(), { x: 0.6, y: 0.3, w: 8, h: 0.28, fontSize: 11, bold: true, color: C.accent3, charSpacing: 3 });
  s.addText(title, { placeholder: "title" });
  return s;
}

const TOP = 1.45, BOTTOM = 6.62; // content band, above the caption line and footer
// A result figure on a card, scaled to fit the box (x, y, w, h) and centred in it; returns the card's box.
async function figure(slide, file, x, y, w, h, alt) {
  const meta = await sharp(path.join(FIG, file)).metadata();
  const pad = 0.12;
  let iw = w - 2 * pad, ih = (iw * meta.height) / meta.width;
  if (ih > h - 2 * pad) { ih = h - 2 * pad; iw = (ih * meta.width) / meta.height; }
  const cw = iw + 2 * pad, ch = ih + 2 * pad, cx = x, cy = y + (h - ch) / 2;
  card(slide, cx, cy, cw, ch, alt + " frame");
  slide.addImage({ path: path.join(FIG, file), x: cx + pad, y: cy + pad, w: iw, h: ih, altText: alt, objectName: alt });
  return { x: cx, y: cy, w: cw, h: ch };
}
function caption(slide, t, x = 0.6, w = 12.1) {
  text(slide, t, { x, y: 6.68, w, h: 0.26, fontSize: 10, color: C.background2, italic: true });
}
// Right-hand insight panel: two headline numbers and the implication for the team.
function insights(slide, x, items, implication) {
  const w = 12.73 - x;
  card(slide, x, TOP, w, BOTTOM - TOP, "insight panel");
  items.forEach(([big, color, body, size = 34], i) => {
    const y = TOP + 0.3 + i * 1.6;
    text(slide, big, { x: x + 0.3, y, w: w - 0.5, h: 0.75, fontSize: size, bold: true, color, fontFace: THEME.headFontFace, valign: "middle" });
    text(slide, body, { x: x + 0.3, y: y + 0.78, w: w - 0.55, h: 0.75, fontSize: 13, color: C.background2 });
  });
  const y = TOP + 3.65;
  slide.addShape(pres.shapes.LINE, { x: x + 0.3, y, w: w - 0.6, h: 0, line: { color: MUTED_LINE, width: 1 }, objectName: "panel divider" });
  text(slide, "SO WHAT", { x: x + 0.3, y: y + 0.18, w: w - 0.6, h: 0.28, fontSize: 11, bold: true, color: C.accent2, charSpacing: 3 });
  text(slide, implication, { x: x + 0.3, y: y + 0.5, w: w - 0.55, h: 0.95, fontSize: 14, bold: true });
}

// pptxgenjs writes missing values as empty <c:v/> points; a missing point is what draws a gap.
async function dropBlankPoints(file) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  for (const name of Object.keys(zip.files).filter(n => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    const xml = await zip.file(name).async("string");
    zip.file(name, xml.replace(/<c:pt idx="\d+"><c:v><\/c:v><\/c:pt>/g, ""));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  // ===== 1. Title =====
  pres.addSection({ title: "Summary" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Summary" });
  // A stylised sun: a solar-yellow disc inside neon purple coronal rings, bleeding off the right edge.
  s.addShape(pres.shapes.OVAL, { x: 8.3, y: 0.9, w: 6.2, h: 6.2, fill: { color: H.dk1 }, line: { color: H.accent1, width: 1.5, transparency: 55 }, objectName: "corona outer" });
  s.addShape(pres.shapes.OVAL, { x: 8.9, y: 1.5, w: 5.0, h: 5.0, fill: { color: H.dk1 }, line: { color: H.accent1, width: 3 }, objectName: "corona ring" });
  s.addShape(pres.shapes.OVAL, { x: 9.6, y: 2.2, w: 3.6, h: 3.6, fill: { color: H.accent2 }, line: { color: H.accent2, width: 0 }, objectName: "sun disc" });
  s.addShape(pres.shapes.OVAL, { x: 10.35, y: 3.05, w: 0.9, h: 1.3, fill: { color: H.dk1, transparency: 25 }, line: { color: H.dk1, width: 0 }, objectName: "coronal hole" });
  s.addShape(pres.shapes.OVAL, { x: 8.15, y: 3.85, w: 0.3, h: 0.3, fill: { color: H.accent3 }, line: { color: H.accent3, width: 0 }, objectName: "ACE at L1" });
  text(s, "ACE · L1", { x: 7.35, y: 4.25, w: 1.9, h: 0.3, fontSize: 11, color: C.accent3, align: "center" });
  text(s, "SOLAR WIND FORECASTING", { x: 0.7, y: 1.45, w: 7, h: 0.4, fontSize: 14, bold: true, color: C.accent3, charSpacing: 4 });
  s.addText("Forecasting the Solar Wind 72 Hours Ahead", { placeholder: "title" });
  s.addText("Results and submission recommendation for our tournament entry", { placeholder: "body" });
  text(s, "FDL BenchLab Solar Wind Prediction Tournament 2026  ·  Results review, 1 November 2026",
    { x: 0.7, y: 6.55, w: 8.5, h: 0.35, fontSize: 12, color: C.accent5 });
  s.addNotes("Results of our entry and the recommendation for the submission: a gradient-boosted hybrid model that forecasts ACE hourly solar-wind speed 72 hours ahead.");

  // ===== 2. Executive summary =====
  s = content("Executive summary", "A hybrid booster beats every baseline, ready to ship", "Summary");
  card(s, 0.6, TOP, 4.7, BOTTOM - TOP, "bottom line card", H.accent3);
  text(s, "BOTTOM LINE", { x: 0.95, y: TOP + 0.35, w: 4, h: 0.3, fontSize: 11, bold: true, color: C.accent3, charSpacing: 3 });
  text(s, "−23%", { x: 0.95, y: TOP + 0.8, w: 4, h: 1.2, fontSize: 66, bold: true, color: C.accent3, fontFace: THEME.headFontFace, valign: "middle" });
  text(s, "error vs persistence", { x: 0.95, y: TOP + 2.0, w: 4, h: 0.4, fontSize: 16, bold: true });
  text(s, "Test RMSE 75.3 km/s vs 97.8 km/s over 4,105 forecasts, Jan 2024 – Oct 2026, never used for training or tuning.",
    { x: 0.95, y: TOP + 2.5, w: 4.05, h: 1.0, fontSize: 13, color: C.background2 });
  s.addShape(pres.shapes.LINE, { x: 0.95, y: TOP + 3.65, w: 4.0, h: 0, line: { color: MUTED_LINE, width: 1 }, objectName: "bottom line divider" });
  text(s, "RECOMMENDATION", { x: 0.95, y: TOP + 3.83, w: 4, h: 0.28, fontSize: 11, bold: true, color: C.accent2, charSpacing: 3 });
  text(s, "Submit the tuned LightGBM hybrid, with persistence for the first ~10 hours.", { x: 0.95, y: TOP + 4.15, w: 4.05, h: 0.85, fontSize: 15, bold: true });
  const findings = [
    ["FaChartLine", H.accent3, "Beats every baseline from 10 h on", "72-hour RMSE of 91 km/s against 129 km/s for persistence"],
    ["FaSun", H.accent1, "Solar images add real skill", "Coronal holes cut RMSE by 3.4 km/s overall and 5.5 km/s at 72 h"],
    ["FaLayerGroup", H.accent2, "The learner is not the lever", "Five boosting libraries land within 2 km/s; tuning adds 0.3"],
    ["FaExclamationTriangle", H.accent6, "One open risk: the metric", "Pooled TIMSE rewards over-forecasting; aggregation to confirm"],
  ];
  for (let i = 0; i < findings.length; i++) {
    const [ic, col, head, body] = findings[i];
    const y = TOP + i * 1.32;
    card(s, 5.55, y, 7.18, 1.16, "finding card " + (i + 1));
    await iconCircle(s, ic, col, 5.8, y + 0.23, 0.7, "finding icon " + (i + 1));
    text(s, head, { x: 6.8, y: y + 0.2, w: 5.7, h: 0.38, fontSize: 17, bold: true });
    text(s, body, { x: 6.8, y: y + 0.6, w: 5.7, h: 0.4, fontSize: 13, color: C.background2 });
  }
  s.addNotes("Four messages: the hybrid clearly beats the baselines; the solar images are worth their complexity; further gains will not come from swapping the learning library; and the official metric must be clarified before optimizing for it.");

  // ===== 3. The challenge =====
  s = content("The challenge", "72-hour forecasts every 6 hours on 15 CPU minutes", "Summary");
  const task = [
    ["72 h", "Forecast horizon", "Hourly ACE solar-wind speed at L1, 72 values per issuance", "FaWaveSquare", H.accent1, C.accent1],
    ["6 h", "Issuance cadence", "A stateless Docker container is run by the organizers", "FaSyncAlt", H.accent2, C.accent2],
    ["15 min", "CPU budget", "Per run, CPU only: the model must be light to evaluate", "FaMicrochip", H.accent3, C.accent3],
    ["TIMSE", "Official metric", "Threshold-integrated MSE in (km/s)³: weights errors in fast wind", "FaBullseye", H.accent1, C.accent1],
  ];
  for (let i = 0; i < 4; i++) {
    const [big, head, body, ic, hex, col] = task[i];
    const x = 0.6 + i * 3.08;
    card(s, x, 1.65, 2.83, 3.9, "task card " + (i + 1));
    await iconCircle(s, ic, hex, x + 0.3, 1.95, 0.7, "task icon " + (i + 1));
    text(s, big, { x: x + 0.3, y: 2.9, w: 2.4, h: 0.85, fontSize: 40, bold: true, color: col, fontFace: THEME.headFontFace });
    text(s, head, { x: x + 0.3, y: 3.8, w: 2.4, h: 0.35, fontSize: 16, bold: true });
    text(s, body, { x: x + 0.3, y: 4.2, w: 2.3, h: 1.2, fontSize: 13, color: C.background2 });
  }
  const steps = [["1 Nov", "Submission (today)"], ["1–10 Nov", "Pre-deployment"], ["11 Nov", "Live scoring starts"]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.OVAL, { x, y: 6.0, w: 0.22, h: 0.22, fill: { color: H.accent2 }, line: { color: H.accent2, width: 0 }, objectName: "milestone " + (i + 1) });
    text(s, [{ text: steps[i][0] + "  ", options: { bold: true, color: C.accent2 } }, { text: steps[i][1], options: { color: C.background2 } }],
      { x: x + 0.35, y: 5.93, w: 3.6, h: 0.36, fontSize: 14, valign: "middle" });
  }
  s.addNotes("Every 6 hours the organizers run our container with the data available at that time. It must return 72 hourly speeds within 15 minutes of CPU time.");

  // ===== 4. Approach =====
  pres.addSection({ title: "Results" });
  s = content("Approach", "One model fuses solar images, rotation and L1 data", "Results");
  const srcs = [
    ["FaSatelliteDish", H.accent3, "L1 plasma (ACE)", "Last speed and its age; 6, 24 and 72 h means; 24 h max and trend; density, temperature"],
    ["FaRedo", H.accent2, "27-day recurrence", "Speed one and two solar rotations before each target hour"],
    ["FaSun", H.accent1, "Coronal holes (AIA 193 Å)", "Dark-area fraction in six 15° longitude strips, images 0, 24 and 48 h old"],
  ];
  for (let i = 0; i < 3; i++) {
    const [ic, col, head, body] = srcs[i];
    const y = 1.55 + i * 1.75;
    card(s, 0.6, y, 5.6, 1.5, "source card " + (i + 1));
    await iconCircle(s, ic, col, 0.85, y + 0.4, 0.7, "source icon " + (i + 1));
    text(s, head, { x: 1.8, y: y + 0.2, w: 4.2, h: 0.4, fontSize: 17, bold: true });
    text(s, body, { x: 1.8, y: y + 0.62, w: 4.2, h: 0.8, fontSize: 13, color: C.background2 });
  }
  for (let i = 0; i < 3; i++) {
    const y1 = 2.3 + i * 1.75, y2 = 4.05; // from the source card's middle to the model card
    s.addShape(pres.shapes.LINE, { x: 6.2, y: Math.min(y1, y2), w: 0.75, h: Math.max(Math.abs(y2 - y1), 0.01), flipV: y1 > y2,
      line: { color: H.accent5, width: 1.5 }, objectName: "flow line " + (i + 1) });
  }
  card(s, 6.95, 3.0, 3.25, 2.1, "model card", H.accent1);
  await iconCircle(s, "FaTree", H.accent1, 7.2, 3.25, 0.6, "model icon");
  text(s, "Gradient-boosted trees", { x: 7.95, y: 3.25, w: 2.15, h: 0.6, fontSize: 15, bold: true, valign: "middle" });
  text(s, "One model for all 72 lead hours\n51 features · 1.15 M training pairs", { x: 7.2, y: 4.0, w: 2.9, h: 0.9, fontSize: 13, color: C.background2 });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 10.35, y: 3.85, w: 0.6, h: 0.4, fill: { color: H.accent3 }, line: { color: H.accent3, width: 0 }, objectName: "flow arrow" });
  card(s, 11.1, 3.0, 1.6, 2.1, "output card", H.accent3);
  text(s, "72 h", { x: 11.1, y: 3.35, w: 1.6, h: 0.7, fontSize: 30, bold: true, color: C.accent3, align: "center", fontFace: THEME.headFontFace });
  text(s, "hourly speed forecast", { x: 11.25, y: 4.1, w: 1.3, h: 0.8, fontSize: 13, color: C.background2, align: "center" });
  text(s, [{ text: "Strictly causal: ", options: { bold: true, color: C.accent2 } },
    { text: "every input stops at t₀ − 1 h, the data latency in operations", options: { color: C.background2 } }],
    { x: 6.95, y: 5.75, w: 5.75, h: 0.5, fontSize: 14 });
  s.addNotes("Each training row is one (issuance, lead hour) pair, with the lead hour as a feature. Train Feb 2011 – Dec 2021, validation 2022–2023, test Jan 2024 – Oct 2026 (appendix). A unit test checks that changing any data after the cutoff leaves the features unchanged.");

  // ===== 5. RMSE by lead =====
  s = content("Accuracy", "The hybrid cuts 72-hour error by 30% vs persistence", "Results");
  let box = await figure(s, "rmse_by_lead.png", 0.6, TOP, 8.7, BOTTOM - TOP, "RMSE by lead time figure");
  caption(s, "Test period Jan 2024 – Oct 2026, 4,105 issuances every 6 h. Hybrid = tuned LightGBM.", 0.6, box.w);
  insights(s, box.x + box.w + 0.25, [
    ["91 vs 129", C.accent3, "km/s RMSE at 72 h: hybrid against persistence"],
    ["10 h", C.accent2, "crossover: before it, persistence wins (21 vs 40 km/s at 1 h)"],
  ], "Use persistence for leads under 10 h in the submission.");
  s.addNotes("Persistence is hard to beat in the first hours because the wind changes slowly; after about 10 hours the hybrid is the best model at every lead. Recurrence is flat around 105–110 km/s.");

  // ===== 6. Case study =====
  s = content("Case study", "The model anticipates a fast-wind stream days ahead", "Results");
  box = await figure(s, "single_forecast.png", 0.6, TOP, 8.95, BOTTOM - TOP, "single forecast figure");
  caption(s, "Forecast issued 3 Feb 2024 18:00 UTC, test period. Labels show the RMSE of this issuance in km/s.", 0.6, box.w);
  insights(s, box.x + box.w + 0.25, [
    ["40 km/s", C.accent3, "hybrid RMSE, against 111 for persistence"],
    ["72 km/s", C.accent1, "same model without the solar images"],
  ], "The coronal hole was visible on the Sun before the stream reached L1.");
  s.addNotes("Chosen as the issuance with the largest coronal-hole gain among fast-wind arrivals; across the 41 such arrivals the median gain is 5 km/s and 66% of them improve. Persistence cannot anticipate the rise; recurrence sees it partly.");

  // ===== 7. Two months =====
  s = content("Track record", "Two months of forecasts track the wind at 24 and 72 h", "Results");
  await figure(s, "fixed_lead_window.png", 0.6, TOP, 12.13, BOTTOM - TOP, "fixed lead window figure");
  caption(s, "1 Feb – 31 Mar 2024, test period. Each point is one 6-hourly issuance verified at a fixed lead time; panel titles give RMSE in km/s.");
  s.addNotes("At 24 h the hybrid follows most stream arrivals; at 72 h it still captures the timing of recurrent streams but underestimates peaks such as 25 March, a transient event that no input announces days ahead.");

  // ===== 8. Coronal holes =====
  s = content("Why it works", "Coronal holes signal fast wind 3 to 7 days ahead", "Results");
  box = await figure(s, "coronal_hole_lags.png", 0.6, TOP, 8.7, BOTTOM - TOP, "coronal hole lag figure");
  caption(s, "Spearman correlation of hole area per 15° longitude strip with later ACE speed, 2011–2026 (east = negative).", 0.6, box.w);
  insights(s, box.x + box.w + 0.25, [
    ["~1 day", C.accent2, "later peak for every 15° strip further east, as the Sun rotates"],
    ["−3.4 km/s", C.accent1, "RMSE from the solar images (95% CI 1.4–5.5); −5.5 at 72 h"],
  ], "The images pay most at long leads, where L1 data say least.");
  s.addNotes("Pixels are projected to heliographic coordinates with area weights. The lag matches solar rotation (13° per day) plus the transit to L1. The gain is consistent across all five libraries (2.1 to 3.4 km/s).");

  // ===== 9. Libraries =====
  s = content("Model choice", "Five boosting libraries land within 2 km/s", "Results");
  const libs = [["Recurrence", 107.8], ["Persistence", 97.8], ["NGBoost", 77.5], ["CatBoost", 76.4], ["LightGBM", 76.3],
    ["scikit-learn HGB", 76.0], ["XGBoost", 75.7], ["LightGBM, tuned", 75.3]];
  card(s, 0.6, TOP, 8.6, BOTTOM - TOP, "library chart frame");
  s.addChart(pres.charts.BAR, [{ name: "Test RMSE (km/s)", labels: libs.map(l => l[0]), values: libs.map(l => l[1]) }],
    chartBase({ x: 0.75, y: TOP + 0.15, w: 8.3, h: BOTTOM - TOP - 0.3, barDir: "bar",
      chartColors: [H.accent5, H.accent5, H.accent1, H.accent1, H.accent1, H.accent1, H.accent1, H.accent3],
      showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 120,
      valAxisMajorUnit: 20, barGapWidthPct: 45, showTitle: true, title: "Test RMSE, Jan 2024 – Oct 2026 (km/s)", titleColor: H.lt1,
      titleFontFace: "+mn-lt", titleFontSize: 13, catAxisLabelFontSize: 12, objectName: "library comparison chart" }));
  caption(s, "Same features, splits and tree size for every library; only LightGBM was tuned (40-trial random search).", 0.6, 8.6);
  insights(s, 9.45, [
    ["≤ 0.7", C.accent2, "km/s between XGBoost, HGB, LightGBM and CatBoost"],
    ["+0.3", C.accent3, "km/s from tuning: 75.6 → 75.3 (95% CI −0.3 to 1.0)"],
  ], "Invest in inputs and the metric, not in the learner.");
  s.addNotes("XGBoost is 0.6 km/s better than default LightGBM (CI 0.1 to 1.1); the other differences are not significant. NGBoost trades 1–2 km/s for a predicted spread and needs 2 hours to train.");

  // ===== 10. TIMSE =====
  pres.addSection({ title: "Risks" });
  s = content("Risk", "Pooled TIMSE rewards over-forecasting", "Risks");
  box = await figure(s, "timse_behavior.png", 0.6, TOP, 8.95, BOTTOM - TOP, "TIMSE behaviour figure");
  caption(s, "Hybrid forecasts on the full test period plus a constant offset; TIMSE pooled over all hours.", 0.6, box.w);
  insights(s, box.x + box.w + 0.25, [
    ["42.1 M", C.accent2, "TIMSE of a constant 700 km/s, better than the hybrid (74.6 M)"],
    ["75–88%", C.accent1, "of the score from thresholds above 700 km/s, 1.7% of hours"],
  ], "Confirm the aggregation; keep the MSE model meanwhile.");
  s.addNotes("TIMSE integrates the tail MSE over speed thresholds. Pooled over the whole period it is dominated by a few extreme events, so pushing forecasts up reduces it. Scored per issuance, the bias stops paying.");

  // ===== 11. Uncertainty =====
  s = content("Uncertainty", "The spread ranks errors, but intervals are too narrow", "Risks");
  box = await figure(s, "ngboost_calibration.png", 0.6, TOP, 8.95, BOTTOM - TOP, "NGBoost calibration figure");
  caption(s, "NGBoost hybrid with a normal distribution per hour, test period Jan 2024 – Oct 2026.", 0.6, box.w);
  insights(s, box.x + box.w + 0.25, [
    ["65%", C.accent2, "of hours fall inside the nominal 95% interval"],
    ["41 → 71", C.accent3, "km/s error from the lowest to the highest spread quintile"],
  ], "Recalibrate the spread on 2022–2023 before use.");
  s.addNotes("The spread is informative but too narrow, especially with coronal-hole features: median spread 27 km/s against a mean absolute error of 53. The model hit the 2000-iteration cap, which likely over-sharpens the scale.");

  // ===== 12. Recommendation =====
  pres.addSection({ title: "Recommendation" });
  s = content("Recommendation", "Submit the tuned hybrid today, then de-risk", "Recommendation");
  card(s, 0.6, TOP, 12.13, 1.35, "decision card", H.accent3);
  await iconCircle(s, "FaCheck", H.accent3, 0.9, TOP + 0.3, 0.75, "decision icon");
  text(s, "DECISION", { x: 1.95, y: TOP + 0.25, w: 10, h: 0.28, fontSize: 11, bold: true, color: C.accent3, charSpacing: 3 });
  text(s, "Ship the tuned LightGBM hybrid in the container, with persistence for leads under 10 h.",
    { x: 1.95, y: TOP + 0.58, w: 10.5, h: 0.5, fontSize: 17, bold: true });
  const plan = [
    ["Today · 1 Nov", H.accent3, C.accent3, "Package and submit", "Replace the starter-kit example; check the run fits the 15-minute CPU budget"],
    ["1–10 Nov", H.accent1, C.accent1, "Verify real-time inputs", "Compare near-real-time AIA features with the archive used for training"],
    ["Before 11 Nov", H.accent2, C.accent2, "Settle the metric", "Ask how TIMSE is aggregated: pooled, per issuance or per window"],
    ["After launch", H.accent5, C.background2, "Add calibrated uncertainty", "Rescale the NGBoost spread on 2022–2023 and test quantile forecasts"],
  ];
  const ty = 3.35;
  s.addShape(pres.shapes.LINE, { x: 0.9, y: ty, w: 11.5, h: 0, line: { color: MUTED_LINE, width: 2 }, objectName: "timeline" });
  plan.forEach(([when, hex, col, head, body], i) => {
    const x = 0.6 + i * 3.08;
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: ty - 0.14, w: 0.28, h: 0.28, fill: { color: hex }, line: { color: hex, width: 0 }, objectName: "timeline dot " + (i + 1) });
    text(s, when, { x: x + 0.6, y: ty - 0.17, w: 2.3, h: 0.34, fontSize: 14, bold: true, color: col, valign: "middle" });
    card(s, x, 3.8, 2.85, 2.82, "plan card " + (i + 1));
    text(s, String(i + 1).padStart(2, "0"), { x: x + 0.3, y: 4.05, w: 1.2, h: 0.6, fontSize: 28, bold: true, color: col, fontFace: THEME.headFontFace });
    text(s, head, { x: x + 0.3, y: 4.75, w: 2.35, h: 0.65, fontSize: 16, bold: true });
    text(s, body, { x: x + 0.3, y: 5.45, w: 2.35, h: 1.05, fontSize: 13, color: C.background2 });
  });
  s.addNotes("The submission pack still holds the starter-kit example, so packaging is the priority today. The other steps reduce risk before and during live scoring.");

  // ===== Appendix A1. Validation =====
  pres.addSection({ title: "Appendix" });
  s = content("Appendix · Validation", "A clean temporal split, with the test period scored once", "Appendix");
  const span = [["Train", "Feb 2011 – Dec 2021", "Fit the trees", 10.9, H.accent1, C.accent1],
    ["Validation", "2022 – 2023", "Pick trees and hyperparameters", 2.0, H.accent2, C.accent2],
    ["Test", "Jan 2024 – Oct 2026", "4,105 issuances, scored once", 2.8, H.accent3, C.accent3]];
  const total = span.reduce((a, b) => a + b[3], 0), W0 = 12.1;
  let x0 = 0.6;
  span.forEach(([name, when, what, yrs, hex, sc], i) => {
    const w = (yrs / total) * W0 - 0.08;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0, y: 1.75, w, h: 0.9, rectRadius: 0.1, fill: { color: hex }, line: { color: hex, width: 0 }, objectName: name + " span" });
    text(s, name, { x: x0 + 0.12, y: 1.75, w: w - 0.14, h: 0.9, fontSize: 15, bold: true, color: C.text1, valign: "middle", fontFace: THEME.headFontFace });
    // Captions stay under their own span so neighbouring captions never overlap.
    const cw = i === 0 ? 2.6 : w, align = i === 2 ? "right" : "left";
    text(s, when, { x: x0, y: 2.8, w: cw, h: 0.35, fontSize: 14, bold: true, color: sc, align });
    text(s, what, { x: x0, y: 3.15, w: cw, h: 0.8, fontSize: 13, color: C.background2, align });
    x0 += w + 0.08;
  });
  const checks = [
    ["FaShieldAlt", "No leakage across splits", "Targets of one split end at least 72 h before the next one starts"],
    ["FaLayerGroup", "Block bootstrap", "95% CIs from 39 blocks of 27 days, one solar rotation each"],
    ["FaDatabase", "Operational inputs", "Real-time product only instead of Level 2: RMSE 76.0 → 76.2 km/s"],
  ];
  for (let i = 0; i < 3; i++) {
    const [ic, head, body] = checks[i];
    const x = 0.6 + i * 4.1;
    card(s, x, 4.25, 3.85, 2.3, "check card " + (i + 1));
    await iconCircle(s, ic, H.accent1, x + 0.3, 4.5, 0.6, "check icon " + (i + 1));
    text(s, head, { x: x + 1.1, y: 4.5, w: 2.6, h: 0.7, fontSize: 16, bold: true, valign: "middle" });
    text(s, body, { x: x + 0.3, y: 5.3, w: 3.3, h: 1.1, fontSize: 13, color: C.background2 });
  }
  s.addNotes("After 9 July 2024 the targets come only from the operational SWEPAM product because Level 2 ends.");

  // ===== Appendix A2. Tuning =====
  s = content("Appendix · Tuning", "A 40-trial search moved validation RMSE by under 1 km/s", "Appendix");
  const trials = D.trials;
  const tColors = trials.map((v, i) => (i === 0 ? H.accent3 : Math.abs(v - D.ref_trial) < 1e-9 ? H.accent2 : H.accent1));
  let refSeen = false; // one reference bar in yellow (first match only)
  for (let i = 0; i < trials.length; i++) if (tColors[i] === H.accent2) { if (refSeen) tColors[i] = H.accent1; refSeen = true; }
  card(s, 0.6, TOP, 8.6, BOTTOM - TOP, "tuning chart frame");
  s.addChart(pres.charts.BAR, [{ name: "Validation RMSE", labels: trials.map((_, i) => String(i + 1)), values: trials }],
    chartBase({ x: 0.75, y: TOP + 0.15, w: 8.3, h: BOTTOM - TOP - 0.3, barDir: "col", chartColors: tColors, showLegend: false,
      valAxisMinVal: 67.5, valAxisMaxVal: 69.25, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00", barGapWidthPct: 35,
      showValAxisTitle: true, valAxisTitle: "Validation RMSE 2022–2023 (km/s)", showCatAxisTitle: true,
      catAxisTitle: "LightGBM trials, sorted from best to worst", catAxisLabelFrequency: 5,
      showTitle: true, title: "Random search over leaves, step, leaf size, L2 and subsampling", titleColor: H.lt1,
      titleFontFace: "+mn-lt", titleFontSize: 13, objectName: "tuning trials chart" }));
  caption(s, "Green: best trial. Yellow: reference configuration. The axis starts at 67.5 km/s.", 0.6, 8.6);
  insights(s, 9.45, [
    ["75.6 → 75.3", C.accent3, "km/s test RMSE; 95% CI of the gain −0.3 to 1.0", 27],
    ["2 h 19 min", C.accent2, "on 32 cores via SLURM; the search resumes after a crash", 27],
  ], "Best: step 0.03, 63 leaves, 500 per leaf, half the features per tree.");
  s.addNotes("34 of 40 trials fall between 67.9 and 68.5 km/s. Choosing trees by RMSE instead of TIMSE already moved the reference from 76.3 to 75.6.");

  await pres.writeFile({ fileName: OUT });
  await dropBlankPoints(OUT);
  await applyTheme(OUT, THEME);
  console.log("wrote", path.resolve(OUT));
})().catch(e => { console.error(e); process.exit(1); });
