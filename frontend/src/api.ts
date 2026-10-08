export type Tensor = { shape: number[]; data: number[] };
export type Weights = Record<string, Tensor>;
export type Acts = Record<string, number[]>;
export type Prediction = { activations: Acts; probs: number[] };

export type HistoryPoint = { step: number; epoch: number; loss: number; acc: number };
export type EpochPoint = { epoch: number; step: number; test_acc?: number; test_loss?: number };

export type ServerMessage =
  | {
      type: "state";
      status: "idle" | "running" | "paused";
      global_step: number;
      history: HistoryPoint[];
      epoch_history: EpochPoint[];
      test_acc: number | null;
    }
  | (HistoryPoint & {
      type: "progress";
      epoch_index: number;
      epochs: number;
      batch: number;
      steps_per_epoch: number;
      weights: Weights;
      prediction: Prediction | null;
    })
  | (EpochPoint & { type: "epoch_end" })
  | {
      type: "model";
      reason: string;
      weights: Weights;
      prediction: Prediction | null;
      eval: { test_acc: number; test_loss: number } | null;
    }
  | { type: "error"; message: string };

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }));
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    throw new Error(detail);
  }
  return r.json();
}

export const get = <T>(path: string) => fetch(path).then((r) => handle<T>(r));

export const post = <T = { ok: boolean }>(path: string, body: unknown = {}) =>
  fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => handle<T>(r));

/** WebSocket that reconnects on drop. */
export function connect(onMessage: (m: ServerMessage) => void, onStatus: (up: boolean) => void): void {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/train`;
  const open = () => {
    const ws = new WebSocket(url);
    ws.onopen = () => onStatus(true);
    ws.onmessage = (e) => onMessage(JSON.parse(e.data));
    ws.onclose = () => {
      onStatus(false);
      setTimeout(open, 1000);
    };
  };
  open();
}
