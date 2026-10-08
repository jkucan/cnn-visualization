# Handwritten digit CNN — live training demo

Train a tiny convolutional network on MNIST (resized to 8×8) and watch every weight and activation change as it learns. Draw a digit and see how each layer responds to it.

- **Randomize weights** (Kaiming / Xavier / Normal / Uniform, with an optional seed), or **load the pretrained** checkpoint (~90% test accuracy; the network is kept small for clarity, not accuracy)
- **Train** with adjustable epochs, learning rate, batch size, optimizer, number of training images and augmentation. You can **pause**, **step one batch** at a time, or **slow training down** to watch it.
- **Draw** on a 280×280 canvas. The server crops the drawing, scales it into a 6 px box and centers it by center of mass in an 8×8 frame, the same way MNIST digits are prepared.
- **Visualization:** the input pixels, conv kernels (blue = negative, red = positive), feature maps before and after pooling, dense-layer edges (weight, or weight × activation) and the 10 output probabilities. Loss and accuracy charts update live.

Architecture: `1×8×8 → Conv 4@3×3 → ReLU → max-pool 2×2 (4×4×4 = 64) → FC 64→16 → ReLU → FC 16→10`. That is 1,250 parameters in total. The sizes are set in `backend/app/config.py` (`INPUT_SIZE`, `CONV_FILTERS`, `HIDDEN`), and the UI lays itself out from them.

## Run

```bash
make build   # production image: builds the frontend, bakes in MNIST and the pretrained model
make run     # http://localhost:8000
```

## Develop

```bash
make dev     # backend with auto-reload (:8000) + Vite dev server (:5173). Open http://localhost:5173
make test    # backend pytest in the dev container
make lock    # regenerate frontend/package-lock.json inside a container
```

On the first `make dev`, MNIST is downloaded into the `data` Docker volume and a pretrained checkpoint is trained. This takes about 10 seconds.

## Stack

- **Backend:** Python 3.12, PyTorch (CPU), FastAPI + Uvicorn, with a WebSocket (`/ws/train`) that streams training progress. API docs are at `/docs`.
- **Frontend:** Vite + TypeScript. The network is drawn with Canvas 2D; the charts, scales and colors use D3.

## Layout

```
backend/app/   main.py (API + WebSocket), trainer.py (background training), model.py,
               data.py (MNIST → 8×8 cache), preprocess.py (canvas → 8×8), pretrain.py
backend/tests/ pytest
frontend/src/  main.ts (controls), draw.ts (canvas), viz/network.ts, viz/charts.ts, viz/colors.ts
```
