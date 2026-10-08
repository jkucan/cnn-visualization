# CNN Handwritten Digit Demo — Plan

An interactive demo that trains a small convolutional neural network (CNN) on handwritten digits and shows what happens inside it: the input pixels, the weights and activations of every layer, and the output probabilities for 0–9. The user can reset the weights to random values, train the network while watching it, and draw their own digits for the network to classify.

---

## 1. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Model and training | **Python 3.12 + PyTorch (CPU build)** | Standard tool. The network is tiny, so the CPU is fast enough and no GPU setup is needed in Docker. |
| Dataset | **torchvision MNIST**, resized 28×28 → 16×16 | 60k training and 10k test images. Downloaded once, when the Docker image is built. |
| Backend API | **FastAPI + Uvicorn** | Async, has WebSocket support built in, and generates API docs automatically at `/docs`. |
| Live updates | **WebSocket** (`/ws/train`) | Streams loss, accuracy and weights to the browser while training runs. |
| Image preprocessing | **NumPy + Pillow** (+ `scipy.ndimage` for center of mass) | Drawn digits go through the same pipeline as MNIST (see §4). |
| Frontend build | **Vite + TypeScript** | Fast dev server, simple production build, no framework needed. |
| Visualization | **Canvas 2D** for heatmaps and weight edges, **D3.js** for scales, color maps and charts | Canvas can draw thousands of edges and pixels quickly. D3 handles color scales and the loss/accuracy charts. |
| UI framework (optional) | Plain TS, or **Svelte** if the UI grows | Keep it minimal. The UI is one page with a few panels. |
| Packaging | **Docker** (multi-stage build) + **docker compose** | Separate dev and production setups (see §7). |
| Tests | **pytest** (backend), **Vitest** (frontend utils), optional **Playwright** end-to-end | |

**Alternative considered:** running everything in the browser with TensorFlow.js, with no backend. That is simpler to host, but it drops PyTorch and makes it harder to reuse the Python training code. Not chosen, but the frontend is designed so this swap stays possible later.

---

## 2. Network architecture (input 1×16×16)

> **Revised:** to make it clearer as a teaching tool, the network is now `8×8 input → one conv layer (4 filters 3×3) → ReLU → max-pool → 16 hidden → 10 outputs`. That is 1,250 parameters and about 90% test accuracy. The sizes are set in `backend/app/config.py`. The rest of this section describes the original 16×16 design.

The network is kept small on purpose, so that **every weight can be shown**. It should still reach about 97–98% test accuracy.

```
Input        1 × 16 × 16                          (256 pixels)
Conv1        4 filters, 3×3, pad 1  → 4 × 16 × 16   ReLU     params: 4·9 + 4      =    40
MaxPool 2×2                          → 4 × 8 × 8
Conv2        8 filters, 3×3, pad 1  → 8 × 8 × 8    ReLU     params: 8·4·9 + 8    =   296
MaxPool 2×2                          → 8 × 4 × 4   (flatten → 128)
FC1          128 → 32                               ReLU     params: 128·32 + 32  = 4,128
FC2 (output) 32 → 10                                Softmax  params: 32·10 + 10   =   330
                                                                          total ≈ 4,794
```

- `forward(x, return_activations=True)` returns a dict with the output of every stage: `input, conv1, pool1, conv2, pool2, fc1, logits, probs`. The visualizer uses this.
- The architecture is defined in a single config (numbers of filters and hidden units), so it is easy to try variants such as 6/12 filters or 64 hidden units.

---

## 3. Features

### 3.1 Randomize weights
- A **"Randomize weights"** button, with options for:
  - initialization: Kaiming-uniform (PyTorch default), Xavier, or plain uniform/normal with a chosen scale
  - random seed, so a run can be repeated
- After a reset, the output probabilities should be roughly uniform (about 10% each). This is the baseline for showing that the untrained net knows nothing.

### 3.2 Training
- Controls: **epochs**, **learning rate**, **batch size**, **optimizer** (SGD / Adam), **training subset size** (for example, only 1k images to make training slow enough to watch), plus **Start / Pause / Step one batch / Stop**.
- Training runs in a background task on the server. Every *N* batches (throttled to about 5–10 updates per second), the server sends over the WebSocket:
  - step, epoch, training loss, running accuracy
  - a snapshot of all weights (about 4.8k floats, which is small enough to send in full)
  - the activations for the user's current drawing, so the prediction **visibly improves while the net trains**
- At the end of each epoch it evaluates on the test set and reports test accuracy.
- A **loss/accuracy chart** plots these values live.
- Optional extra: **"Add my drawing to the training set"** with a label, to show fine-tuning on the user's own handwriting.
- Save and load checkpoints to a Docker volume. Also ship a **pretrained checkpoint**, so the demo can jump straight to "trained".

### 3.3 Drawing canvas
- A **280×280** canvas: mouse and touch input, adjustable brush size (default about 18 px), a **Clear** button, and live prediction while drawing (debounced to about 100 ms).
- Also a **"Random test sample"** button that loads an MNIST test image into the pipeline, for comparison with the user's drawing.

### 3.4 Visualization panels (left → right)
1. **Input layer**: a 16×16 grid, one cell per pixel, with the value shown as grayscale and in a tooltip. This is the image after preprocessing, so it looks like what the network actually sees.
2. **Conv1**:
   - 4 kernels, each 3×3, drawn with a diverging color map (blue for negative, white for 0, red for positive)
   - 4 feature maps (16×16) after ReLU, drawn with a sequential color map, and the 8×8 pooled versions
   - Hovering a feature-map pixel highlights its 3×3 receptive field in the input.
3. **Conv2**:
   - 8 filters, each 4×3×3, shown as a small 8×4 grid of 3×3 tiles
   - 8 feature maps (8×8) and the pooled versions (4×4)
4. **FC1 (32 nodes)**:
   - Each node is a circle whose brightness is its activation.
   - There are 128 → 32 = 4,096 edges. By default only the **top-k edges by |weight × input activation|** are drawn, so the view stays readable. A toggle shows all edges, with transparency based on |w| and color based on the sign of w.
   - Hovering a node shows its incoming weights reshaped as an 8×4×4 image, so you can see which patterns it responds to.
5. **Output (10 nodes)**:
   - 32 → 10 edges, all drawn, using the same color and transparency coding.
   - 10 nodes labeled 0–9, each with a **probability bar** and percentage. The most likely digit is highlighted.
- **Legend** for the color maps, plus a switch to show **weights**, **activations**, or both.
- Optional: a confusion matrix on the test set after each epoch.

---

## 4. Preprocessing (important for accuracy on drawn digits)

MNIST digits are size-normalized and centered. Raw drawings are not, and a model trained on MNIST will do badly on them unless both go through the same steps. One function, `preprocess.py`, handles both:

1. Convert the canvas image to grayscale, invert if needed (white digit on black background, like MNIST), and scale values to 0–1.
2. Crop to the **bounding box** of the ink. If the canvas is empty, return a blank image.
3. Resize, keeping the aspect ratio, so the longest side is **12 px**. This matches MNIST's 20/28 ratio at 16 px. Use antialiasing (Pillow `LANCZOS` or `BOX`).
4. Paste onto a 16×16 black image, **centered by center of mass**.
5. Normalize with the MNIST mean and std (computed on the 16×16 version).

For training data, MNIST is resized 28 → 16 with antialiasing (this is already centered, so steps 2–4 are not needed). Resized tensors are cached to `data/mnist16.pt` at build time, so training starts instantly.

---

## 5. Backend API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/model/reset` | `{init, seed}` → reset weights to random, return the new weights |
| `GET`  | `/api/model/weights` | Full weight snapshot |
| `GET`  | `/api/model/info` | Architecture description (layers, shapes). Lets the frontend lay out panels without hardcoding them. |
| `POST` | `/api/predict` | `{image: base64 PNG}` → `{input16, activations, probs}` |
| `GET`  | `/api/samples?n=1&split=test` | Random MNIST sample(s), with their labels |
| `POST` | `/api/train/start` | `{epochs, lr, batch_size, optimizer, subset}` |
| `POST` | `/api/train/pause` · `/resume` · `/step` · `/stop` | Training control |
| `POST` | `/api/model/save` · `/load` | Checkpoints in `/app/checkpoints` |
| `WS`   | `/ws/train` | Stream of `{type: "progress" \| "epoch_end" \| "done", ...}` |

**Concurrency:** training runs in a worker thread, and a lock protects the model. Each optimizer step and each `/predict` forward pass holds the lock, so predictions never read weights while training is halfway through an update. (A single-image forward pass takes microseconds, so there is no need to copy the weights.)

---

## 6. Project layout

```
cnn-visualization/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, routes, WebSocket, serves static frontend
│   │   ├── model.py         # SmallCNN + activation capture + init schemes
│   │   ├── data.py          # MNIST download, 16×16 resize, caching, samples
│   │   ├── preprocess.py    # canvas → 16×16 tensor (§4)
│   │   ├── trainer.py       # background training loop, pause/step/stop, progress events
│   │   └── schemas.py       # Pydantic request/response models
│   ├── tests/               # pytest: shapes, preprocess, API, training smoke test
│   ├── checkpoints/pretrained.pt
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── main.ts
│   │   ├── api.ts           # REST + WebSocket client
│   │   ├── draw/canvas.ts   # drawing pad
│   │   ├── viz/heatmap.ts   # reusable grid renderer (input, kernels, feature maps)
│   │   ├── viz/dense.ts     # nodes + edges renderer for FC layers
│   │   ├── viz/output.ts    # probability bars
│   │   ├── viz/charts.ts    # loss/accuracy charts (D3)
│   │   ├── viz/colormaps.ts
│   │   └── controls.ts      # randomize / train panels
│   ├── package.json
│   └── vite.config.ts       # proxies /api and /ws to backend in dev
├── Dockerfile               # multi-stage production image
├── docker-compose.yml       # dev: backend (reload) + frontend (vite)
├── docker-compose.prod.yml  # prod: single image
├── Makefile                 # make dev | build | run | test
└── README.md
```

---

## 7. Docker: develop, package, run

### Development (`make dev` → `docker compose up`)
- **backend** service: `python:3.12-slim` with the CPU PyTorch wheel (`--index-url https://download.pytorch.org/whl/cpu`). The source is bind-mounted and run with `uvicorn --reload` on port 8000. Named volumes hold `data/` (MNIST) and `checkpoints/`.
- **frontend** service: `node:22-alpine` running `vite --host` on port 5173, with the source bind-mounted. Vite forwards `/api` and `/ws` to `backend:8000`.
- Open `http://localhost:5173`. Code changes reload automatically on both sides.

### Production image (`make build`)
Multi-stage `Dockerfile`:
1. **Stage `frontend-build`** (`node:22-alpine`): `npm ci && npm run build` → `dist/`
2. **Stage `runtime`** (`python:3.12-slim`):
   - install the CPU PyTorch wheel and the requirements
   - copy the backend code, then `RUN python -m app.data --prepare` to download MNIST and cache the 16×16 tensors inside the image, so it works offline
   - copy `dist/` to `/app/static`. FastAPI serves it with `StaticFiles`.
   - run as a non-root user, `EXPOSE 8000`, add a `HEALTHCHECK` on `/api/model/info`
   - `CMD uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Expected image size is about 900 MB, mostly the CPU PyTorch wheel. (If size matters later, the server could run inference only with ONNX Runtime, but training needs PyTorch.)

### Run (`make run`)
```
docker run --rm -p 8000:8000 -v cnn-checkpoints:/app/checkpoints cnn-digits-demo
```
Open `http://localhost:8000`.

**Apple Silicon:** the image builds natively for arm64. Inside Docker the Mac GPU (MPS) is not available, which is fine, because one epoch of this network on 60k 16×16 images takes only a few seconds on the CPU. Use `docker buildx build --platform linux/amd64,linux/arm64` if the image will be shared.

---

## 8. Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | **Scaffold + Docker dev setup** | `make dev` runs both services, and the frontend gets a reply from `/api/model/info` |
| 2 | **Model + data + training (headless)** | pytest passes. Training on 1 epoch of the full set gives >95% test accuracy. |
| 3 | **Preprocess + predict API** | Drawn and sample images come back as 16×16 images plus activations. Unit tests cover empty, off-center and tiny drawings. |
| 4 | **Drawing canvas + input + output panels** | Drawing a digit shows the 16×16 input and live probability bars (using the pretrained checkpoint) |
| 5 | **Conv layer visualization** | Kernels and feature maps render, and the receptive-field hover works |
| 6 | **Dense layer visualization** | FC1 and FC2 nodes and edges render, with top-k filtering and node-hover weight images |
| 7 | **Randomize + live training over WebSocket** | Reset → about 10% for each digit. Start training → loss chart, weight colors and predictions update live. Pause, step and stop work. |
| 8 | **Production image + polish** | `make build && make run` serves the full app from one container. README, legends and tooltips are in place. |
| 9 | *(Optional)* extras | Add-my-drawing fine-tuning, confusion matrix, architecture variants, Playwright end-to-end test |

---

## 9. Risks and mitigations

- **Drawn digits get classified wrongly** because they don't look like MNIST. Fix: use the preprocessing in §4, a thick brush, and a small amount of augmentation during training (random shifts of ±1 px and small rotations).
- **Too many edges to read** (4,096 in FC1). Fix: show only the top-k by default, highlight on hover, and draw on Canvas instead of SVG.
- **WebSocket flooding** during fast training. Fix: send updates at most every ~100 ms and drop intermediate snapshots.
- **Training and prediction touching the model at the same time.** Fix: a lock plus predicting from a snapshot copy of the weights.
- **Training too fast to watch.** Fix: the training-subset-size control, a small default learning rate for demos, and "step one batch" mode.
