# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run locally (Windows):**
```bat
run.bat          # auto-crea venv, instala deps, abre http://127.0.0.1:5000
```

**Run manually (any OS):**
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**Production (gunicorn):**
```bash
gunicorn app:app --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:$PORT
```
Use **1 worker** only — the inference engine (ENGINE singleton) lives in memory; multiple workers would each have their own copy that could diverge after a retrain.

**Skip model bootstrap** (faster startup for dev/testing):
```bash
JNUS_NO_WARM=1 python app.py
```

**Verify the API is up:**
```
GET http://127.0.0.1:5000/api/health
```

## Architecture

There are **two independent ML systems** in one Flask app that must not be confused:

### 1. Consumer Inference Engine (`engine.py` → `JanusEngine` singleton)
This is the **production FinTech product**. It:
- Trains 4 models once on startup (or loads from `models/janus_bundle.pkl`)
- Serves the public app at `/app` via `POST /api/score`
- Never exposes training controls to end users
- The bundle structure: `{models, scaler, columns, ranges, metrics, raw_num, raw_cat, approval_rate, version, dataset_size, trained_at, source}`

The synthetic seed dataset (`generate_seed()`) represents Ecuador credit data. Features: `edad`, `ingresos_mensuales`, `cargas_familiares`, `creditos_activos`, `sexo`, `educacion`, `historial_pagos`, `institucion`, `tipo_credito`, `situacion_laboral` → target `aprobado`.

### 2. Data Science Pipeline (`app.py` STATE dict)
This is the **original research pipeline**, kept intact. It flows through:
`POST /api/upload` → `POST /api/preprocess` → `POST /api/train` → `GET /api/predict`

State lives in a global `STATE` dict (single-user, in-memory). These endpoints are NOT used by the consumer app — they exist for the hidden data-science workflow.

### Route Map

| Route | Access | Serves |
|-------|--------|--------|
| `/` | public | redirects → `/app` |
| `/app` | public | `janus_app.html` — consumer FinTech app |
| `/lab` | public | redirects → `/app` (legacy alias) |
| `/admin` | **login required** | `admin.html` — model management dashboard |
| `/admin/login` | public | login form |
| `/api/score` | public | consumer inference (engine.py) |
| `/api/options` | public | dropdown values for the form |
| `/api/model_info` | public | read-only model metadata |
| `/api/admin/retrain` | **admin only** | upload dataset → retrain → hot-reload |
| `/api/admin/retrain_seed` | **admin only** | regenerate demo model |
| `/api/upload`, `/api/preprocess`, `/api/train`, `/api/predict` | public | legacy data-science pipeline |

### Admin Authentication

`admin_required` decorator uses Flask session (`session['is_admin']`). Credentials come from env vars:
- `JNUS_ADMIN_USER` (default: `admin`)
- `JNUS_ADMIN_PASSWORD` (default: `jnus2026`)
- `JNUS_SECRET_KEY` (default: insecure dev key — **must set in production**)

### Admin → App Model Update Flow

```
Admin uploads CSV/SAV/XLSX → POST /api/admin/retrain
  → retrain_from_dataframe() in engine.py
  → train_and_persist() trains 4 models with 80/20 split for honest metrics,
    then re-fits on full dataset for production
  → joblib.dump() → models/janus_bundle.pkl
  → ENGINE.bundle = bundle  ← hot-reload: /app immediately uses new model
```

The `.pkl` file is the contract between admin training and consumer inference. It is gitignored and auto-generated on first startup.

### XAI (Explainable AI)

Two XAI implementations exist:

**Consumer app (`engine.py`):** Uses logistic regression coefficients for `positive_factors`/`negative_factors`, plus real what-if analysis in `_recommendations()` — modifies payload values and recomputes ensemble probability.

**Data science pipeline (`app.py` `/api/predict`):** Per-model attribution. For models without `coef_`/`feature_importances_` (MLP), uses ablation: set each feature to its median, measure probability delta `prob - prob_ablated`.

### Frontend

Single template: `templates/janus_app.html`. Pure HTML/CSS/JS with no JS framework. Uses an inline SVG sprite for icons (Lucide-style, ids like `i-home`, `i-cpu`, etc.).

UI category → backend model mapping is done **entirely in JS** inside `janus_app.html` — the `CATEGORIES` array maps UI subcategories to `tipo_credito` values the model understands. The backend never sees the UI category names.

PWA: Service worker at `/sw.js` (served from `static/`, scope `/`). Cache name is `janus-ai-v4` — bump this constant when changing cached assets. App shell caches `/app` + 5 PNG assets; API routes are always network-first.

### Key Constraints

- **Do not modify** `/api/upload`, `/api/preprocess`, `/api/train`, `/api/predict`, `engine.py` models/hyperparameters, or XAI logic without understanding both ML systems.
- `models/janus_bundle.pkl` is gitignored. It regenerates automatically at startup; never commit it.
- `safe_jsonable()` in `app.py` must wrap all numpy/pandas values before `jsonify()` — numpy types are not JSON-serializable.
- The `_encode()` function in `engine.py` uses `drop_first=False` (unlike the pipeline which uses `drop_first=True`) — this is intentional so the column set is stable and predictable for inference.
- When retraining from an admin-uploaded dataset, all 11 columns in `REQUIRED_COLUMNS` must be present or the upload is rejected with a clear error listing what's missing.

## Deployment

Deploy to Render with one click via `render.yaml` (Blueprint). Set `JNUS_SECRET_KEY`, `JNUS_ADMIN_USER`, `JNUS_ADMIN_PASSWORD` in the Render dashboard environment panel (they are `sync: false` — not stored in the yaml).

For Capacitor APK packaging: `capacitor.config.json` points `server.url` to the live domain (`https://janus.siac.ai/app`). The mobile shell is `capacitor-www/index.html` (fallback only).
