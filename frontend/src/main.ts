import { connect, get, post, type EpochPoint, type HistoryPoint, type Prediction, type ServerMessage } from "./api";
import { DrawPad } from "./draw";
import { LineChart } from "./viz/charts";
import { SERIES_1, SERIES_2, actColor, cssGradient, weightColor } from "./viz/colors";
import { NetworkView } from "./viz/network";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const val = (id: string) => $<HTMLInputElement>(id).value;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

// ---------------------------------------------------------------- state

let status: "idle" | "running" | "paused" = "idle";
let history: HistoryPoint[] = [];
let epochHistory: EpochPoint[] = [];
// Test accuracy of freshly reset/loaded weights: the step-0 point of the test series.
let baseline: EpochPoint | null = null;
let connected = false;

// The layout is built from the server's architecture description.
type Info = { input_size: number; layers: { name: string; shape: number[] }[] };
const info = await get<Info>("/api/model/info");
const layer = (name: string) => info.layers.find((l) => l.name === name)!.shape;
const dims = { size: info.input_size, filters: layer("conv1")[0], hidden: layer("fc1")[0], classes: layer("fc2")[0] };
const blank = () => new Array(dims.size * dims.size).fill(0);

const net = new NetworkView($<HTMLCanvasElement>("net"), $("tooltip"), dims);
const topk = $<HTMLInputElement>("topk");
topk.max = String(net.denseEdgeCount);
topk.step = "16";
topk.value = String(net.topK);
$("topk-val").textContent = `showing the strongest ${net.topK} of ${net.denseEdgeCount}`;
const lossChart = new LineChart($("loss-chart"), { yFormat: (v) => v.toFixed(2), emptyText: "Train to see the loss fall" });
const accChart = new LineChart($("acc-chart"), { yFormat: (v) => `${Math.round(v * 100)}%`, yDomain: [0, 1], emptyText: "No training yet" });

$("lg-weight").style.background = cssGradient(weightColor, -1, 1);
$("lg-act").style.background = cssGradient(actColor, 0, 1);

function showMessage(text: string | null) {
  const el = $("message");
  el.hidden = !text;
  el.textContent = text ?? "";
  if (text) setTimeout(() => { if (el.textContent === text) el.hidden = true; }, 6000);
}

async function run(fn: () => Promise<unknown>) {
  try {
    await fn();
  } catch (e) {
    showMessage((e as Error).message);
  }
}

function applyPrediction(p: Prediction | null) {
  if (p) net.setActivations(p.activations);
}

// ---------------------------------------------------------------- UI updates

function renderStatus() {
  const pill = $("status");
  pill.className = `pill ${connected ? status : ""}`;
  pill.textContent = connected ? status : "disconnected";
  const idle = status === "idle", running = status === "running", paused = status === "paused";
  const set = (id: string, enabled: boolean) => { $<HTMLButtonElement>(id).disabled = !enabled || !connected; };
  set("randomize", true); // stops training first if needed
  set("load-pretrained", idle);
  set("load-user", idle);
  set("save", idle);
  set("start", true); // Train / Pause / Resume toggle
  set("step", !running);
  set("stop", !idle);
  $("start").textContent = running ? "Pause" : paused ? "Resume" : "Train";
  for (const id of ["epochs", "lr", "batch", "optimizer", "subset", "augment", "init", "seed", "scale"]) {
    $<HTMLInputElement>(id).disabled = !idle;
  }
}

function renderCharts() {
  lossChart.update([
    { key: "loss", label: "Training loss", color: SERIES_1, points: history.map((p) => ({ x: p.step, y: p.loss })) },
  ]);
  accChart.update([
    { key: "train", label: "Train (recent batches)", color: SERIES_1, points: history.map((p) => ({ x: p.step, y: p.acc })) },
    {
      key: "test", label: "Test set (10k)", color: SERIES_2, markers: true,
      points: [...(baseline ? [baseline] : []), ...epochHistory]
        .filter((e) => e.test_acc !== undefined)
        .map((e) => ({ x: e.step, y: e.test_acc! })),
    },
  ]);
}

function renderTestAcc(acc: number | null | undefined) {
  $("test-acc").textContent = acc == null ? "" : `test accuracy ${pct(acc)}`;
}

// ---------------------------------------------------------------- server events

function onMessage(m: ServerMessage) {
  switch (m.type) {
    case "state":
      status = m.status;
      history = m.history;
      epochHistory = m.epoch_history;
      renderTestAcc(m.test_acc);
      renderStatus();
      renderCharts();
      if (status === "idle") $("progress-text").textContent = history.length ? `done · ${history.at(-1)!.step} steps` : "idle";
      break;
    case "progress": {
      net.setWeights(m.weights);
      applyPrediction(m.prediction);
      history.push({ step: m.step, epoch: m.epoch, loss: m.loss, acc: m.acc });
      renderCharts();
      const done = (m.epoch_index * m.steps_per_epoch + m.batch) / (m.epochs * m.steps_per_epoch);
      $("progress-bar").style.width = pct(done);
      $("progress-text").textContent =
        `epoch ${m.epoch_index + 1}/${m.epochs} · batch ${m.batch}/${m.steps_per_epoch} · loss ${m.loss.toFixed(3)} · acc ${pct(m.acc)}`;
      break;
    }
    case "epoch_end":
      epochHistory.push(m);
      renderTestAcc(m.test_acc);
      renderCharts();
      break;
    case "model":
      net.setWeights(m.weights);
      applyPrediction(m.prediction);
      if (m.reason !== "connect") {
        history = [];
        epochHistory = [];
        baseline = m.eval ? { epoch: 0, step: 0, test_acc: m.eval.test_acc } : null;
        $("progress-bar").style.width = "0";
        $("progress-text").textContent = m.reason === "reset" ? "weights randomized" : `loaded ${m.reason} weights`;
        renderTestAcc(m.eval?.test_acc);
        renderCharts();
      }
      break;
    case "error":
      showMessage(`Training error: ${m.message}`);
      break;
  }
}

connect(onMessage, (up) => {
  connected = up;
  renderStatus();
  if (up) predictPixels(blank());
});

// ---------------------------------------------------------------- drawing

let pending: Promise<unknown> | null = null;
let queued: (() => Promise<Prediction>) | null = null;

/** Keep at most one predict request in flight; newest request wins. */
function schedulePredict(req: () => Promise<Prediction>) {
  if (pending) {
    queued = req;
    return;
  }
  pending = req()
    .then((p) => applyPrediction(p))
    .catch((e) => showMessage(e.message))
    .finally(() => {
      pending = null;
      const next = queued;
      queued = null;
      if (next) schedulePredict(next);
    });
}

const predictPixels = (pixels: number[]) => schedulePredict(() => post<Prediction>("/api/predict", { pixels }));

const pad = new DrawPad($<HTMLCanvasElement>("pad"), (dataUrl) => {
  $("sample-label").textContent = "";
  if (dataUrl) schedulePredict(() => post<Prediction>("/api/predict", { image: dataUrl }));
  else predictPixels(blank());
});

$("clear").onclick = () => pad.clear();
$("brush").oninput = () => { pad.brush = Number(val("brush")); };
$("sample").onclick = () => run(async () => {
  const [s] = await get<{ label: number; pixels: number[] }[]>("/api/samples?n=1&split=test");
  pad.showPixels(s.pixels);
  $("sample-label").textContent = `true label: ${s.label}`;
  predictPixels(s.pixels);
  setDrawer(false); // show the loaded digit
});

// ---------------------------------------------------------------- weights

$("init").onchange = () => {
  $("scale-wrap").hidden = !["normal", "uniform"].includes(val("init"));
};
$("randomize").onclick = () => run(async () => {
  if (status !== "idle") await post("/api/train/stop");
  await post("/api/model/reset", {
    init: val("init"),
    seed: val("seed") === "" ? null : Number(val("seed")),
    scale: Number(val("scale")) || 0.1,
  });
});
$("load-pretrained").onclick = () => run(() => post("/api/model/load", { name: "pretrained" }));
$("load-user").onclick = () => run(() => post("/api/model/load", { name: "user" }));
$("save").onclick = () => run(async () => {
  await post("/api/model/save", { name: "user" });
  showMessage(null);
  $("progress-text").textContent = "checkpoint saved";
});

// ---------------------------------------------------------------- training

const trainConfig = (paused = false) => ({
  epochs: Number(val("epochs")),
  lr: Number(val("lr")),
  batch_size: Number(val("batch")),
  optimizer: val("optimizer"),
  subset: Number(val("subset")),
  augment: $<HTMLInputElement>("augment").checked,
  delay_ms: Number(val("delay")),
  paused,
});

// One button: Train -> Pause -> Resume.
$("start").onclick = () => run(() =>
  status === "running" ? post("/api/train/pause")
    : status === "paused" ? post("/api/train/resume")
      : post("/api/train/start", trainConfig()));
$("stop").onclick = () => run(() => post("/api/train/stop"));
$("step").onclick = () => run(async () => {
  if (status === "idle") await post("/api/train/start", trainConfig(true));
  await post("/api/train/step");
});
$("delay").oninput = () => {
  $("delay-val").textContent = `${val("delay")} ms / batch`;
  if (status !== "idle") run(() => post("/api/train/speed", { delay_ms: Number(val("delay")) }));
};

// ---------------------------------------------------------------- network view controls

$("edge-mode").onchange = () => {
  net.edgeMode = val("edge-mode") as "contribution" | "weight";
  net.invalidate();
};
$("topk").oninput = () => {
  net.topK = Number(val("topk"));
  $("topk-val").textContent = `showing the strongest ${val("topk")} of ${net.denseEdgeCount}`;
  net.invalidate();
};

// ---------------------------------------------------------------- settings drawer

const drawer = $("settings"), backdrop = $("backdrop"), opener = $("open-settings");

function setDrawer(open: boolean) {
  drawer.classList.toggle("open", open);
  drawer.inert = !open;
  drawer.setAttribute("aria-hidden", String(!open));
  backdrop.hidden = !open;
  opener.setAttribute("aria-expanded", String(open));
  if (open) $("close-settings").focus();
  else opener.focus();
}

opener.onclick = () => setDrawer(true);
$("close-settings").onclick = () => setDrawer(false);
backdrop.onclick = () => setDrawer(false);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawer.classList.contains("open")) setDrawer(false);
});

renderStatus();
renderCharts();
