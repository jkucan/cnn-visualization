IMAGE := cnn-digits-demo

.PHONY: dev down build run test lock

dev:            ## backend (reload) + Vite dev server -> http://localhost:5173
	docker compose up --build

down:
	docker compose down

build:          ## production image
	docker build --target runtime -t $(IMAGE) .

run:            ## run production image -> http://localhost:8000
	docker run --rm -p 8000:8000 -v cnn-checkpoints:/checkpoints $(IMAGE)

test:           ## backend tests (uses the dev image and the MNIST data volume)
	docker compose run --rm --no-deps backend sh -c "python -m app.data --prepare && pytest -q"

lock:           ## regenerate frontend/package-lock.json without installing on the host
	docker run --rm -v $(PWD)/frontend:/app -w /app node:22-alpine npm install --package-lock-only
