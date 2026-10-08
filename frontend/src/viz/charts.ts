import * as d3 from "d3";
import { GUIDE, SURFACE, TEXT_MUTED, TEXT_SECONDARY } from "./colors";

export type Pt = { x: number; y: number };
export type Series = { key: string; label: string; color: string; points: Pt[]; markers?: boolean };

const M = { top: 8, right: 44, bottom: 20, left: 38 };
const HEIGHT = 120;

/** Small line chart with a crosshair tooltip. One y-axis only. */
export class LineChart {
  private svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private legend: HTMLElement;
  private tip: HTMLElement;
  private series: Series[] = [];

  constructor(
    private root: HTMLElement,
    private opts: { yFormat: (v: number) => string; yDomain?: [number, number]; emptyText: string },
  ) {
    this.legend = root.appendChild(document.createElement("div"));
    this.legend.className = "legend";
    this.svg = d3.select(root).append("svg").attr("height", HEIGHT);
    this.tip = root.appendChild(document.createElement("div"));
    this.tip.className = "chart-tip";
    this.tip.hidden = true;
    new ResizeObserver(() => this.render()).observe(root);
  }

  update(series: Series[]) {
    this.series = series;
    this.render();
  }

  private render() {
    const width = this.root.clientWidth;
    if (!width) return;
    const svg = this.svg.attr("width", width);
    svg.selectAll("*").remove();

    // Legend only when there is more than one series.
    this.legend.innerHTML = this.series.length > 1
      ? this.series.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("")
      : "";

    const all = this.series.flatMap((s) => s.points);
    const iw = width - M.left - M.right, ih = HEIGHT - M.top - M.bottom;
    const g = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    if (!all.length) {
      g.append("text").attr("x", iw / 2).attr("y", ih / 2).attr("text-anchor", "middle")
        .attr("fill", TEXT_MUTED).attr("font-size", 12).text(this.opts.emptyText);
      return;
    }

    const x = d3.scaleLinear().domain([0, d3.max(all, (p) => p.x) || 1]).range([0, iw]);
    const y = d3.scaleLinear()
      .domain(this.opts.yDomain ?? [0, (d3.max(all, (p) => p.y) || 1) * 1.05])
      .range([ih, 0]).nice();

    // Recessive grid + axes.
    g.append("g").selectAll("line").data(y.ticks(4)).join("line")
      .attr("x1", 0).attr("x2", iw).attr("y1", y).attr("y2", y).attr("stroke", GUIDE).attr("stroke-width", 1);
    g.append("g").selectAll("text").data(y.ticks(4)).join("text")
      .attr("x", -6).attr("y", y).attr("dy", "0.32em").attr("text-anchor", "end")
      .attr("fill", TEXT_MUTED).attr("font-size", 10).text((d) => this.opts.yFormat(d));
    g.append("g").selectAll("text").data(x.ticks(4).filter(Number.isInteger)).join("text")
      .attr("x", x).attr("y", ih + 14).attr("text-anchor", "middle")
      .attr("fill", TEXT_MUTED).attr("font-size", 10).text((d) => d3.format("~s")(d));
    g.append("text").attr("x", iw).attr("y", ih + 14).attr("dx", 6).attr("fill", TEXT_MUTED)
      .attr("font-size", 10).text("step");

    const line = d3.line<Pt>().x((p) => x(p.x)).y((p) => y(p.y));
    const labelYs: number[] = [];
    for (const s of this.series) {
      if (!s.points.length) continue;
      if (s.points.length > 1 && !s.markers) {
        g.append("path").datum(s.points).attr("d", line).attr("fill", "none")
          .attr("stroke", s.color).attr("stroke-width", 2).attr("stroke-linejoin", "round");
      }
      if (s.markers) {
        if (s.points.length > 1) {
          g.append("path").datum(s.points).attr("d", line).attr("fill", "none")
            .attr("stroke", s.color).attr("stroke-width", 2).attr("stroke-dasharray", "2 3");
        }
        g.append("g").selectAll("circle").data(s.points).join("circle")
          .attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y)).attr("r", 4)
          .attr("fill", s.color).attr("stroke", SURFACE).attr("stroke-width", 2);
      }
      // Direct label: latest value at the line's end.
      // Nudge apart from earlier labels so they never overlap.
      const last = s.points[s.points.length - 1];
      let ly = y(last.y);
      for (const other of labelYs) if (Math.abs(ly - other) < 11) ly = other + (ly <= other ? -11 : 11);
      labelYs.push(ly);
      g.append("text").attr("x", x(last.x) + 6).attr("y", ly).attr("dy", "0.32em")
        .attr("fill", TEXT_SECONDARY).attr("font-size", 10).text(this.opts.yFormat(last.y));
    }

    // Crosshair + tooltip.
    const cross = g.append("line").attr("y1", 0).attr("y2", ih).attr("stroke", TEXT_MUTED).attr("visibility", "hidden");
    const dots = this.series.map((s) =>
      g.append("circle").attr("r", 4).attr("fill", s.color).attr("stroke", SURFACE).attr("stroke-width", 2).attr("visibility", "hidden"));
    const bisect = d3.bisector<Pt, number>((p) => p.x).center;
    g.append("rect").attr("width", iw).attr("height", ih).attr("fill", "transparent")
      .on("mousemove", (ev: MouseEvent) => {
        const [mx] = d3.pointer(ev);
        const xv = x.invert(mx);
        const rows: string[] = [];
        let cx = mx;
        this.series.forEach((s, i) => {
          if (!s.points.length) return;
          const p = s.points[bisect(s.points, xv)];
          if (i === 0) cx = x(p.x);
          dots[i].attr("cx", x(p.x)).attr("cy", y(p.y)).attr("visibility", "visible");
          rows.push(`<div><i style="background:${s.color}"></i>${s.label}: <b>${this.opts.yFormat(p.y)}</b> <span class="muted">@ ${p.x}</span></div>`);
        });
        cross.attr("x1", cx).attr("x2", cx).attr("visibility", "visible");
        this.tip.innerHTML = rows.join("");
        this.tip.hidden = false;
        const left = M.left + cx + 12;
        this.tip.style.left = `${Math.min(left, width - this.tip.offsetWidth - 4)}px`;
        this.tip.style.top = `${M.top + 10 + this.legend.offsetHeight}px`;
      })
      .on("mouseleave", () => {
        cross.attr("visibility", "hidden");
        dots.forEach((d) => d.attr("visibility", "hidden"));
        this.tip.hidden = true;
      });
  }
}
