/** 280x280 drawing pad: white ink on black, like MNIST. */
export class DrawPad {
  private ctx: CanvasRenderingContext2D;
  private drawing = false;
  private last: { x: number; y: number } | null = null;
  private timer = 0;
  private showingSample = false;
  brush = 20;

  constructor(
    private canvas: HTMLCanvasElement,
    private onChange: (dataUrl: string | null) => void,
  ) {
    this.ctx = canvas.getContext("2d")!;
    this.clear(false);
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    for (const ev of ["pointerup", "pointercancel", "pointerleave"]) {
      canvas.addEventListener(ev, () => this.up());
    }
  }

  clear(notify = true) {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.showingSample = false;
    if (notify) this.onChange(null);
  }

  /** Show a square sample (row-major 0..1 values), upscaled. */
  showPixels(pixels: number[]) {
    const n = Math.round(Math.sqrt(pixels.length));
    const small = document.createElement("canvas");
    small.width = small.height = n;
    const sctx = small.getContext("2d")!;
    const img = sctx.createImageData(n, n);
    pixels.forEach((v, i) => {
      const c = Math.round(v * 255);
      img.data.set([c, c, c, 255], i * 4);
    });
    sctx.putImageData(img, 0, 0);
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(small, 0, 0, this.canvas.width, this.canvas.height);
    this.showingSample = true;
  }

  private pos(e: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * this.canvas.width,
      y: ((e.clientY - r.top) / r.height) * this.canvas.height,
    };
  }

  private down(e: PointerEvent) {
    if (this.showingSample) this.clear(false);
    this.canvas.setPointerCapture(e.pointerId);
    this.drawing = true;
    this.last = this.pos(e);
    this.stroke(this.last);
  }

  private move(e: PointerEvent) {
    if (!this.drawing) return;
    this.stroke(this.pos(e));
  }

  private up() {
    if (!this.drawing) return;
    this.drawing = false;
    this.last = null;
    this.emit(0);
  }

  private stroke(p: { x: number; y: number }) {
    const ctx = this.ctx;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = this.brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(this.last!.x, this.last!.y);
    ctx.lineTo(p.x + 0.01, p.y);
    ctx.stroke();
    this.last = p;
    this.emit(80);
  }

  /** Debounced: send the drawing at most every `delay` ms while drawing. */
  private emit(delay: number) {
    if (this.timer) {
      if (delay > 0) return;
      clearTimeout(this.timer);
    }
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      this.onChange(this.canvas.toDataURL("image/png"));
    }, delay);
  }
}
