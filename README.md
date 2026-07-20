# EncryptSight

EncryptSight is a desktop-friendly network-flow analyst tool. It scores CSV uploads, keeps prior runs, and exposes flow-level SHAP explanations.

## Two independent lanes

- **UNSW-NB15**: a 10-class LightGBM model. Input must use the UNSW-NB15 feature schema; it produces an attack family and a binary benign/attack label.
- **CICIDS2017**: a separate binary LightGBM model. Input must use CICIDS2017 feature names; it produces benign/attack only.

The lanes are not interchangeable: a CICIDS CSV cannot be sent to the UNSW model, or vice versa.

## Web mode

Copy `.env.example` to `.env`, then run `docker compose up --build`. The UI is available at `http://localhost:5173` and the API at `http://localhost:8000`.

For frontend development: copy `frontend/.env.example` to `frontend/.env`, run `npm.cmd install` and `npm.cmd run dev` in `frontend/`. The API base URL is always read from `VITE_API_BASE_URL`.

## Desktop mode

Install Rust and the Tauri CLI, build the backend sidecar from `backend/sidecar.py` (PyInstaller), name it for Tauri's target triple under `desktop/src-tauri/binaries/`, then run `tauri build` from `desktop/src-tauri`. Tauri bundles `frontend/dist` and starts the FastAPI sidecar on launch.

`backend/app/db/database.py` first attempts `DATABASE_URL` when it is PostgreSQL and otherwise falls back to a local SQLite file. The backend assets are resolved from source-relative paths, which also avoids working-directory differences in the desktop sidecar.

## Models and retraining

Artifacts live in `backend/data/models/`: `etf_lgbm_model.txt` for UNSW-NB15 and `cicids_lgbm_model.txt` for CICIDS2017. The UNSW lane is shipped pretrained. Start CICIDS retraining with `POST /api/models/cicids2017/train`; the resulting active model is recorded in the database. Avoid moving the model files without updating the configured artifact paths.
