import type { Acts, Weights } from "../api";
import {
  GUIDE, SURFACE, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY, WEIGHT_NEG, WEIGHT_POS,
  actColor, inputColor, weightColor,
} from "./colors";

/** Network dimensions, from /api/model/info. */
export interface Dims {
  size: number; // input is size x size
  filters: number; // conv filters (3x3)
  hidden: number; // dense hidden units
  classes: number;
}

// Logical drawing size; the canvas is scaled to fit its container.
const W = 1220;
const H = 640;
const TOP = 70;
const BOTTOM = 20;

type GridKind = "input" | "kernel" | "conv" | "pool";

interface Grid {
  kind: GridKind;
  ch: number; // filter / map index
  x: number;
  y: number;
  cell: number;
  n: number; // rows = cols
}

interface NodePos {
  layer: "hidden" | "out";
  i: number;
  x: number;
  y: number;
  r: number;
}

type Hover =
  | { kind: "cell"; grid: Grid; r: number; c: number }
  | { kind: "node"; node: NodePos }
  | null;

export type EdgeMode = "contribution" | "weight";

const maxAbs = (a: ArrayLike<number>, from = 0, to = a.length) => {
  let m = 0;
  for (let i = from; i < to; i++) m = Math.max(m, Math.abs(a[i]));
  return m || 1e-9;
};

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));

export class NetworkView {
  private ctx: CanvasRenderingContext2D;
  private grids: Grid[] = [];
  private nodes: NodePos[] = [];
  private weights: Weights | null = null;
  private acts: Acts | null = null;
  private hover: Hover = null;
  private scale = 1;
  private frame = 0;
  private cols = { input: 0, kernel: 0, conv: 0, pool: 0, hidden: 0, out: 0 };
  private readonly pooled: number;

  edgeMode: EdgeMode = "contribution";
  topK = 200;

  constructor(private canvas: HTMLCanvasElement, private tooltip: HTMLElement, private dims: Dims) {
    this.ctx = canvas.getContext("2d")!;
    this.pooled = dims.size / 2;
    this.layout();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement!);
    canvas.addEventListener("mousemove", (e) => this.onMove(e));
    canvas.addEventListener("mouseleave", () => {
      this.hover = null;
      this.tooltip.hidden = true;
      this.invalidate();
    });
    this.resize();
  }

  /** Number of edges between the pooled maps and the hidden layer. */
  get denseEdgeCount() {
    return this.dims.filters * this.pooled * this.pooled * this.dims.hidden;
  }

  setWeights(w: Weights) {
    this.weights = w;
    this.invalidate();
  }

  setActivations(a: Acts | null) {
    this.acts = a;
    this.invalidate();
    if (this.hover) this.updateTooltip();
  }

  invalidate() {
    if (!this.frame) this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.render();
    });
  }

  // ---------------------------------------------------------------- layout

  private layout() {
    const { size: S, filters: F, hidden: Hn, classes: C } = this.dims;
    const P = this.pooled;
    const avail = H - TOP - BOTTOM;

    // Target pixel sizes, shrunk if the filter rows would not fit vertically.
    let mapPx = 96, kernelCell = 16, poolPx = 72, gap = 40;
    const need = F * mapPx + (F - 1) * gap;
    if (need > avail) {
      const k = avail / need;
      mapPx *= k; kernelCell *= k; poolPx *= k; gap *= k;
    }
    const mapCell = mapPx / S, poolCell = poolPx / P;
    const inCell = Math.min(25, 200 / S);

    const c = this.cols;
    c.input = 20;
    c.kernel = 290;
    c.conv = c.kernel + 3 * kernelCell + 36;
    c.pool = c.conv + mapPx + 36;
    c.hidden = 830;
    c.out = 1010;

    const grids: Grid[] = [];
    grids.push({ kind: "input", ch: 0, x: c.input, y: TOP + (avail - S * inCell) / 2, cell: inCell, n: S });
    const top = TOP + (avail - (F * mapPx + (F - 1) * gap)) / 2;
    for (let f = 0; f < F; f++) {
      const y = top + f * (mapPx + gap);
      grids.push({ kind: "kernel", ch: f, x: c.kernel, y: y + (mapPx - 3 * kernelCell) / 2, cell: kernelCell, n: 3 });
      grids.push({ kind: "conv", ch: f, x: c.conv, y, cell: mapCell, n: S });
      grids.push({ kind: "pool", ch: f, x: c.pool, y: y + (mapPx - poolPx) / 2, cell: poolCell, n: P });
    }
    this.grids = grids;

    const nodes: NodePos[] = [];
    const column = (layer: NodePos["layer"], n: number, x: number, maxGap: number, r: number) => {
      const g = n > 1 ? Math.min(maxGap, avail / (n - 1)) : 0;
      const y0 = TOP + (avail - (n - 1) * g) / 2;
      for (let i = 0; i < n; i++) nodes.push({ layer, i, x, y: y0 + i * g, r: Math.min(r, g / 2 - 2 || r) });
    };
    column("hidden", Hn, c.hidden, 34, 10);
    column("out", C, c.out, 56, 16);
    this.nodes = nodes;
  }

  private resize() {
    const width = this.canvas.parentElement!.clientWidth;
    const dpr = window.devicePixelRatio || 1;
    this.scale = width / W;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${H * this.scale}px`;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(H * this.scale * dpr);
    this.ctx.setTransform(dpr * this.scale, 0, 0, dpr * this.scale, 0, 0);
    this.invalidate();
  }

  // ---------------------------------------------------------------- values

  /** Value and color for one grid cell. Returns null when there is nothing to show. */
  private cell(g: Grid, r: number, c: number): { value: number; color: string } | null {
    const w = this.weights, a = this.acts;
    const S = this.dims.size, P = this.pooled;
    switch (g.kind) {
      case "input": {
        if (!a) return null;
        const v = a.input[r * S + c];
        return { value: v, color: inputColor(v) };
      }
      case "kernel": {
        if (!w) return null;
        const d = w["conv1.weight"].data;
        const v = d[g.ch * 9 + r * 3 + c];
        return { value: v, color: weightColor(v / maxAbs(d)) };
      }
      case "pool": {
        // While hovering a hidden unit, the pooled maps show that unit's incoming weights.
        const h = this.hover;
        if (w && h?.kind === "node" && h.node.layer === "hidden") {
          const d = w["fc1.weight"].data;
          const n = this.dims.filters * P * P, from = h.node.i * n;
          const v = d[from + g.ch * P * P + r * P + c];
          return { value: v, color: weightColor(v / maxAbs(d, from, from + n)) };
        }
        if (!a) return null;
        const v = a.pool1[g.ch * P * P + r * P + c];
        return { value: v, color: actColor(v / maxAbs(a.conv1)) };
      }
      case "conv": {
        if (!a) return null;
        const v = a.conv1[g.ch * S * S + r * S + c];
        return { value: v, color: actColor(v / maxAbs(a.conv1)) };
      }
    }
  }

  // ---------------------------------------------------------------- render

  private render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = SURFACE;
    ctx.fillRect(0, 0, W, H);

    this.drawHeaders();
    this.drawGuides();
    this.drawDenseEdges(); // under the grids, so edges appear to leave each pooled cell
    for (const g of this.grids) this.drawGrid(g);
    this.drawReceptiveField();
    this.drawNodes();
  }

  private text(s: string, x: number, y: number, color = TEXT_SECONDARY, size = 12, align: CanvasTextAlign = "left", weight = 400) {
    const ctx = this.ctx;
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(s, x, y);
  }

  private drawHeaders() {
    const { size: S, filters: F, hidden: Hn, classes: C } = this.dims;
    const P = this.pooled;
    const c = this.cols;
    const h = (title: string, sub: string, x: number, align: CanvasTextAlign = "left") => {
      this.text(title, x, 22, TEXT_PRIMARY, 13, align, 600);
      this.text(sub, x, 40, TEXT_MUTED, 11, align);
    };
    const k = this.find("kernel", 0), m = this.find("conv", 0), p = this.find("pool", 0);
    h("Input", `${S}×${S} pixels`, c.input);
    h("Convolution", `${F} filters 3×3 → ReLU → max-pool 2×2`, c.kernel);
    this.text("filter", k.x + (3 * k.cell) / 2, 58, TEXT_MUTED, 10, "center");
    this.text(`feature map ${S}×${S}`, m.x + (S * m.cell) / 2, 58, TEXT_MUTED, 10, "center");
    this.text(`pooled ${P}×${P}`, p.x + (P * p.cell) / 2, 58, TEXT_MUTED, 10, "center");
    h("Hidden", `${Hn} units, ReLU`, c.hidden, "center");
    h("Output", `${C} digits, softmax`, c.out, "center");
  }

  private box(g: Grid) {
    return { x: g.x, y: g.y, w: g.n * g.cell, h: g.n * g.cell };
  }

  private find(kind: GridKind, ch: number) {
    return this.grids.find((g) => g.kind === kind && g.ch === ch)!;
  }

  private line(x1: number, y1: number, x2: number, y2: number, color: string, alpha = 1, width = 1) {
    const ctx = this.ctx;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  private drawGuides() {
    const inp = this.box(this.find("input", 0));
    for (let f = 0; f < this.dims.filters; f++) {
      const k = this.box(this.find("kernel", f));
      const m = this.box(this.find("conv", f));
      const p = this.box(this.find("pool", f));
      this.line(inp.x + inp.w + 4, inp.y + inp.h / 2, k.x - 4, k.y + k.h / 2, GUIDE);
      this.line(k.x + k.w + 4, k.y + k.h / 2, m.x - 4, m.y + m.h / 2, GUIDE);
      this.line(m.x + m.w + 4, m.y + m.h / 2, p.x - 4, p.y + p.h / 2, GUIDE);
    }
  }

  private drawGrid(g: Grid) {
    const ctx = this.ctx;
    const gap = g.cell >= 8 ? 2 : 1;
    const { x, y, w, h } = this.box(g);
    for (let r = 0; r < g.n; r++) {
      for (let c = 0; c < g.n; c++) {
        const cell = this.cell(g, r, c);
        ctx.fillStyle = cell ? cell.color : "#222220";
        ctx.fillRect(x + c * g.cell, y + r * g.cell, g.cell - gap, g.cell - gap);
      }
    }
    const hovered = this.hover?.kind === "cell" && this.hover.grid === g;
    ctx.strokeStyle = hovered ? TEXT_SECONDARY : "#2e2e2b";
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 2, y - 2, w + 2, h + 2);
  }

  /** Highlight the cells a hovered cell depends on in the previous layer. */
  private drawReceptiveField() {
    const h = this.hover;
    if (h?.kind !== "cell") return;
    const ctx = this.ctx;
    const outline = (g: Grid, r0: number, c0: number, size: number) => {
      const r1 = Math.max(0, r0), c1 = Math.max(0, c0);
      const r2 = Math.min(g.n, r0 + size), c2 = Math.min(g.n, c0 + size);
      ctx.strokeStyle = TEXT_PRIMARY;
      ctx.lineWidth = 2;
      ctx.strokeRect(g.x + c1 * g.cell - 1, g.y + r1 * g.cell - 1, (c2 - c1) * g.cell, (r2 - r1) * g.cell);
    };
    const { grid: g, r, c } = h;
    outline(g, r, c, 1);
    if (g.kind === "conv") {
      outline(this.find("input", 0), r - 1, c - 1, 3);
      outline(this.find("kernel", g.ch), 0, 0, 3);
    }
    if (g.kind === "pool") outline(this.find("conv", g.ch), r * 2, c * 2, 2);
  }

  private poolCenter(k: number) {
    const P = this.pooled;
    const g = this.find("pool", Math.floor(k / (P * P)));
    const r = Math.floor((k % (P * P)) / P), c = k % P;
    return { x: g.x + (c + 0.5) * g.cell, y: g.y + (r + 0.5) * g.cell };
  }

  private drawDenseEdges() {
    const w = this.weights;
    if (!w) return;
    const a = this.acts;
    const h = this.hover?.kind === "node" ? this.hover.node : null;
    const hidden = this.nodes.filter((n) => n.layer === "hidden");
    const out = this.nodes.filter((n) => n.layer === "out");
    const useAct = this.edgeMode === "contribution" && a !== null;
    const nIn = this.dims.filters * this.pooled * this.pooled;

    // pooled maps -> hidden
    const w1 = w["fc1.weight"].data;
    const edges1: { j: number; k: number; v: number }[] = [];
    for (let j = 0; j < hidden.length; j++) {
      if (h?.layer === "hidden" && h.i !== j) continue;
      for (let k = 0; k < nIn; k++) {
        const v = w1[j * nIn + k] * (useAct ? a!.pool1[k] : 1);
        if (v !== 0) edges1.push({ j, k, v });
      }
    }
    edges1.sort((p, q) => Math.abs(q.v) - Math.abs(p.v));
    const shown1 = h?.layer === "hidden" ? edges1 : edges1.slice(0, this.topK);
    const m1 = shown1.length ? Math.abs(shown1[0].v) : 1;
    const dim1 = h?.layer === "out" ? 0.25 : 1;
    for (let i = shown1.length - 1; i >= 0; i--) {
      const e = shown1[i];
      const from = this.poolCenter(e.k), to = hidden[e.j];
      const t = Math.abs(e.v) / m1;
      this.line(from.x, from.y, to.x - to.r - 1, to.y, e.v > 0 ? WEIGHT_POS : WEIGHT_NEG, dim1 * (0.06 + 0.8 * t), 0.5 + 1.5 * t);
    }

    // hidden -> out: all edges.
    const w2 = w["fc2.weight"].data;
    const nH = hidden.length;
    const edges2: { j: number; o: number; v: number }[] = [];
    for (let o = 0; o < out.length; o++) {
      for (let j = 0; j < nH; j++) {
        const v = w2[o * nH + j] * (useAct ? a!.fc1[j] : 1);
        if (v !== 0) edges2.push({ j, o, v });
      }
    }
    const m2 = maxAbs(edges2.map((e) => e.v));
    edges2.sort((p, q) => Math.abs(p.v) - Math.abs(q.v));
    for (const e of edges2) {
      let focus = 1;
      if (h?.layer === "out") focus = h.i === e.o ? 1 : 0.1;
      if (h?.layer === "hidden") focus = h.i === e.j ? 1 : 0.1;
      const from = hidden[e.j], to = out[e.o];
      const t = Math.abs(e.v) / m2;
      this.line(from.x + from.r + 1, from.y, to.x - to.r - 1, to.y, e.v > 0 ? WEIGHT_POS : WEIGHT_NEG, focus * (0.06 + 0.8 * t), 0.5 + 2 * t);
    }
  }

  private drawNodes() {
    const ctx = this.ctx;
    const a = this.acts;
    const hMax = a ? maxAbs(a.fc1) : 1;
    const probs = a?.probs;
    const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
    const hovered = this.hover?.kind === "node" ? this.hover.node : null;

    for (const n of this.nodes) {
      const v = n.layer === "hidden" ? (a ? a.fc1[n.i] / hMax : 0) : (probs ? probs[n.i] : 0);
      const strong = n === hovered || (n.layer === "out" && n.i === best);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = a ? actColor(v) : "#222220";
      ctx.fill();
      ctx.lineWidth = strong ? 2 : 1;
      ctx.strokeStyle = strong ? TEXT_PRIMARY : "#55554f";
      ctx.stroke();

      if (n.layer !== "out") continue;
      this.text(String(n.i), n.x, n.y + 0.5, v > 0.55 ? "#0b0b0b" : TEXT_PRIMARY, 15, "center", 700);

      // Probability bar with a faint full-scale track.
      const bx = n.x + n.r + 12, bw = 100, bh = 10;
      ctx.fillStyle = "#2a2a28";
      ctx.beginPath();
      ctx.roundRect(bx, n.y - bh / 2, bw, bh, 4);
      ctx.fill();
      if (probs) {
        ctx.fillStyle = n.i === best ? "#4fd1a0" : "#199e70";
        ctx.beginPath();
        ctx.roundRect(bx, n.y - bh / 2, Math.max(2, bw * v), bh, [0, 4, 4, 0]);
        ctx.fill();
        this.text(`${(v * 100).toFixed(1)}%`, bx + bw + 8, n.y, n.i === best ? TEXT_PRIMARY : TEXT_SECONDARY, 12, "left", n.i === best ? 700 : 400);
      }
    }
  }

  // ---------------------------------------------------------------- hover

  private onMove(e: MouseEvent) {
    const rect = this.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / this.scale;
    const y = (e.clientY - rect.top) / this.scale;
    let hover: Hover = null;

    for (const n of this.nodes) {
      if (Math.hypot(n.x - x, n.y - y) <= n.r + 3) hover = { kind: "node", node: n };
    }
    if (!hover) {
      for (const g of this.grids) {
        const c = Math.floor((x - g.x) / g.cell), r = Math.floor((y - g.y) / g.cell);
        if (r >= 0 && r < g.n && c >= 0 && c < g.n) hover = { kind: "cell", grid: g, r, c };
      }
    }
    const prev = this.hover;
    const same = hover === prev || (hover !== null && prev !== null && (
      (hover.kind === "node" && prev.kind === "node" && hover.node === prev.node) ||
      (hover.kind === "cell" && prev.kind === "cell" && hover.grid === prev.grid && hover.r === prev.r && hover.c === prev.c)));
    this.hover = hover;
    this.canvas.style.cursor = hover ? "crosshair" : "default";
    if (!same) this.invalidate();
    this.updateTooltip(e.clientX, e.clientY);
  }

  private tipX = 0;
  private tipY = 0;

  private updateTooltip(cx = this.tipX, cy = this.tipY) {
    this.tipX = cx;
    this.tipY = cy;
    const h = this.hover;
    if (!h) {
      this.tooltip.hidden = true;
      return;
    }
    const w = this.weights, a = this.acts;
    let html = "";
    if (h.kind === "cell") {
      const { grid: g, r, c } = h;
      const cell = this.cell(g, r, c);
      const names: Record<GridKind, string> = {
        input: "Input pixel",
        kernel: `Filter ${g.ch + 1} weight`,
        conv: `Feature map ${g.ch + 1} (after ReLU)`,
        pool: `Feature map ${g.ch + 1}, pooled`,
      };
      const hint: Record<GridKind, string> = {
        input: "",
        kernel: "The filter slides over the input. Each map value is the sum of filter × input over a 3×3 area, plus the bias.",
        conv: "Boxes show the 3×3 input area and the filter used to compute this value",
        pool: "Max of the 2×2 box in the feature map. Feeds every hidden unit.",
      };
      const bias = w && g.kind === "kernel" ? `<div class="muted">filter bias ${fmt(w["conv1.bias"].data[g.ch])}</div>` : "";
      html = `<b>${names[g.kind]}</b><div>row ${r}, col ${c}: <b>${cell ? fmt(cell.value) : "—"}</b></div>${bias}` +
        (hint[g.kind] ? `<div class="muted">${hint[g.kind]}</div>` : "");
    } else {
      const n = h.node;
      if (n.layer === "hidden") {
        const b = w?.["fc1.bias"].data[n.i];
        html = `<b>Hidden unit ${n.i + 1}</b><div>activation <b>${a ? fmt(a.fc1[n.i]) : "—"}</b></div>` +
          (b !== undefined ? `<div class="muted">bias ${fmt(b)}</div>` : "") +
          `<div class="muted">The pooled maps now show this unit's ${this.dims.filters * this.pooled ** 2} input weights</div>`;
      } else {
        const b = w?.["fc2.bias"].data[n.i];
        html = `<b>Output “${n.i}”</b><div>probability <b>${a ? (a.probs[n.i] * 100).toFixed(2) + "%" : "—"}</b></div>` +
          `<div class="muted">logit ${a ? fmt(a.logits[n.i]) : "—"}${b !== undefined ? ` · bias ${fmt(b)}` : ""}</div>`;
      }
    }
    this.tooltip.innerHTML = html;
    this.tooltip.hidden = false;
    const tw = this.tooltip.offsetWidth;
    const left = cx + 16 + tw > window.innerWidth ? cx - tw - 12 : cx + 16;
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${cy + 14}px`;
  }
}
