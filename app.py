"""
JNUS Credit AI Lab — Backend Flask
====================================
API REST para análisis de riesgo crediticio con IA + econometría.

Endpoints:
  GET  /                  → redirige a /app (frontend único: janus_app.html)
  GET  /app               → app consumidor JNUS AI (ÚNICO frontend)
  GET  /api/health        → ping de salud
  POST /api/upload        → carga CSV/Excel y devuelve metadatos
  POST /api/preprocess    → limpieza + codificación de variables
  POST /api/train         → entrena el modelo seleccionado y devuelve métricas
  POST /api/reset         → limpia el estado de la sesión

Modelos soportados: logit, probit, random_forest, xgboost, neural_net
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import time
import traceback
import warnings

# El scaler se ajusta con nombres de columnas pero en predicción recibe arrays;
# silenciamos solo ese aviso cosmético de sklearn.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, url_for)

# ──────────────────────────────────────────────────────────────────────────────
# SCIENCE STACK (importes tolerantes — si falla xgboost o tensorflow, seguimos)
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm  # Probit clásico
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# Motor de inferencia del producto consumidor (capa nueva, no toca el pipeline)
from engine import ENGINE, BUNDLE_PATH, retrain_from_dataframe, REQUIRED_COLUMNS, PIPELINE_STEPS

# Capa de base de datos (SQLite) — persiste encuestas y resultados para que el
# modelo aprenda con más datos. No toca los modelos ni el pipeline existente.
import db as jdb

from functools import wraps
from flask import session


APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB

# ── Sesión y credenciales (fail-closed en producción · cyber-neo CN-001/CN-005) ──
# "Producción" = despliegue real en la nube. El launcher de escritorio pone
# JNUS_LOCAL=1 → NO es producción (corre sobre HTTP local, sin secretos externos).
IS_LOCAL = os.environ.get("JNUS_LOCAL") == "1"
IS_PROD = (os.environ.get("FLASK_ENV") == "production") and not IS_LOCAL
app.secret_key = os.environ.get("JNUS_SECRET_KEY")
ADMIN_USER = os.environ.get("JNUS_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("JNUS_ADMIN_PASSWORD")
if IS_PROD and (not app.secret_key or not ADMIN_PASSWORD):
    raise RuntimeError(
        "Faltan JNUS_SECRET_KEY / JNUS_ADMIN_PASSWORD en producción. "
        "Configúralos en el panel del host (ver .env.example).")
# En desarrollo: secreto aleatorio efímero (no uno quemado) y credenciales por defecto.
app.secret_key = app.secret_key or os.urandom(32)
ADMIN_PASSWORD = ADMIN_PASSWORD or "jnus2026"

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PROD,
    PERMANENT_SESSION_LIFETIME=_dt.timedelta(days=30),  # "mantener sesión"
    # Recargar plantillas al cambiarlas aunque debug esté off (dev/local/preview).
    TEMPLATES_AUTO_RELOAD=True,
)
app.jinja_env.auto_reload = True


def admin_required(fn):
    """Protege rutas/endpoints de administración. Sin sesión → 401 o redirect."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "No autorizado. Inicia sesión como administrador."}), 401
            return redirect(url_for("admin_login"))
        return fn(*args, **kwargs)
    return wrapper

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO EN MEMORIA (single-user). Para multiusuario usar Flask-Session/DB.
# ──────────────────────────────────────────────────────────────────────────────
STATE: Dict[str, Any] = {
    "df_raw": None,         # pd.DataFrame original
    "df_clean": None,       # tras preprocesado
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_train_raw": None,
    "X_test_raw": None,
    "scaler": None,
    "feature_names": None,
    "target": None,
    "categorical_cols": None,
    "numeric_cols": None,
    "median_map": None,     # para imputar Juan / form
    "mode_map": None,
    "model": None,          # objeto sklearn entrenado
    "sm_model": None,       # statsmodels (Probit)
    "model_name": None,
    "use_scaled": None,
    "feature_ranges": None, # min/max por feature para sliders del form
    "log": [],              # logs paso a paso
}


def log(msg: str) -> None:
    STATE["log"].append({"t": time.strftime("%H:%M:%S"), "msg": msg})


def reset_state() -> None:
    for k in list(STATE.keys()):
        STATE[k] = [] if k == "log" else None


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────
def safe_jsonable(obj: Any) -> Any:
    """Convierte numpy/pandas a tipos JSON-serializables."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [safe_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_jsonable(x) for x in obj]
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Detecta columnas numéricas y categóricas de forma robusta."""
    numeric, categorical = [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            # Si tiene pocos valores únicos y son enteros, podría ser categórico
            unique = s.dropna().unique()
            if len(unique) <= 2 and set(unique).issubset({0, 1, 0.0, 1.0}):
                categorical.append(col)  # binaria
            else:
                numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def normalize_binary_text(series: pd.Series) -> pd.Series:
    """Convierte 'Sí/No', 'Yes/No', 'True/False' a 1/0 cuando aplica."""
    mapping = {
        "sí": 1, "si": 1, "yes": 1, "y": 1, "true": 1, "1": 1, "verdadero": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "falso": 0,
    }
    def conv(v):
        if pd.isna(v):
            return np.nan
        key = str(v).strip().lower()
        return mapping.get(key, v)
    out = series.map(conv)
    # Si tras la conversión todo es 0/1/NaN, la consideramos numérica binaria
    vals = out.dropna().unique()
    if len(vals) and set(vals).issubset({0, 1}):
        return out.astype(float)
    return series


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE PREDICCIÓN (reutilizados por /api/predict y el análisis XAI what-if)
# ──────────────────────────────────────────────────────────────────────────────
def _row_to_input(row) -> np.ndarray:
    """Convierte una fila cruda (lista en orden de feature_names) al espacio de
    entrada del modelo (escalado si el modelo lo requiere)."""
    x_raw = np.array(row, dtype=float).reshape(1, -1)
    if STATE.get("use_scaled") and STATE.get("scaler") is not None:
        return STATE["scaler"].transform(x_raw)
    return x_raw


def _compute_prob(x_in: np.ndarray) -> float:
    """Probabilidad P(Y=1) para un vector ya en el espacio de entrada del modelo."""
    sm_m = STATE.get("sm_model")
    if sm_m is not None:
        x_sm = sm.add_constant(x_in, has_constant="add")
        return float(sm_m.predict(x_sm)[0])
    m = STATE.get("model")
    if hasattr(m, "predict_proba"):
        return float(m.predict_proba(x_in)[0, 1])
    z = float(m.decision_function(x_in)[0])
    return 1.0 / (1.0 + float(np.exp(-z)))


# ──────────────────────────────────────────────────────────────────────────────
# RUTAS · UN SOLO FRONTEND: janus_app.html (producto consumidor JNUS AI)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Single source of truth: la raíz siempre lleva a la app JNUS.
    return redirect(url_for("consumer_app"))


@app.route("/app")
def consumer_app():
    # ÚNICO frontend activo de la aplicación.
    return render_template("janus_app.html")


# Rutas legacy retiradas → redirigen a la app única (evita abrir el dashboard viejo).
@app.route("/lab")
def legacy_lab():
    return redirect(url_for("consumer_app"))


@app.route("/metodologia")
def metodologia():
    """Página PÚBLICA e interactiva que explica la estructura matemática y los
    algoritmos (logit → XGBoost → RF → red GELU+Sigmoide) para que un economista
    o el tutor entiendan el funcionamiento. No requiere login."""
    return render_template("metodologia.html")


@app.route("/api/methodology")
def api_methodology():
    """Artefactos NO sensibles del modelo activo para la página de metodología
    (público): métricas, curvas de pérdida, pesos/signos del logit, importancia,
    arquitectura de la red y prueba de neuronas muertas."""
    try:
        if not ENGINE.ready():
            ENGINE.bootstrap()
        b = ENGINE.bundle or {}
        return jsonify({
            "ok": True,
            "learning": b.get("learning") or {},
            "metrics": b.get("metrics", {}),
            "dataset_size": b.get("dataset_size"),
            "version": b.get("version"),
            "approval_rate": round(b.get("approval_rate", 0) * 100, 1),
            "class_distribution": b.get("class_distribution"),
            "n_features": b.get("n_features"),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN · plataforma privada de administración (login + entreno + versiones)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        user = (request.form.get("user") or "").strip()
        pwd = request.form.get("password") or ""
        if user == ADMIN_USER and pwd == ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin.html", login=True, error="Credenciales incorrectas")
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    return render_template("admin.html", login=True, error=None)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    return render_template("admin.html", login=False, error=None,
                           required=REQUIRED_COLUMNS)


# ─── API ADMIN (protegida) ──────────────────────────────────────────────────
@app.route("/api/admin/retrain", methods=["POST"])
@admin_required
def api_admin_retrain():
    """Sube CSV/Excel/SAV → reentrena el motor de producción → /app usa el nuevo
    modelo automáticamente (hot-reload del bundle)."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo."}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Archivo sin nombre."}), 400
        ext = f.filename.lower().rsplit(".", 1)[-1]
        raw = f.read()

        # Leer según formato (CSV / Excel / SAV)
        if ext == "csv":
            df = None
            for sep in [",", ";", "\t", "|"]:
                for enc in ["utf-8", "latin-1", "utf-8-sig"]:
                    try:
                        cand = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, engine="python")
                        if cand.shape[1] > 1:
                            df = cand
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
            if df is None:
                df = pd.read_csv(io.BytesIO(raw))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(raw))
        elif ext == "sav":
            try:
                import pyreadstat
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                df, _meta = pyreadstat.read_sav(tmp_path)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            except ImportError:
                return jsonify({"error": "Para archivos .SAV instala: pip install pyreadstat"}), 500
        else:
            return jsonify({"error": f"Formato no soportado: .{ext}. Usa CSV, Excel o SAV."}), 400

        df.columns = [str(c).strip() for c in df.columns]
        if df.empty:
            return jsonify({"error": "El archivo está vacío."}), 400

        # Validar esquema requerido
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({
                "error": "Faltan columnas requeridas: " + ", ".join(missing),
                "required": REQUIRED_COLUMNS,
                "received": list(df.columns),
            }), 400

        # Reentrenar el motor de producción + hot-reload
        bundle = retrain_from_dataframe(df, source=f.filename)
        metrics = bundle.get("metrics", {})
        best = max(metrics.items(), key=lambda kv: kv[1].get("auc", 0)) if metrics else (None, {})
        names = {"logit": "Regresión Logística", "random_forest": "Random Forest",
                 "xgboost": "XGBoost", "neural_net": "Red Neuronal"}
        return jsonify({
            "ok": True,
            "message": "Modelo reentrenado y publicado. La app pública ya usa el nuevo modelo.",
            "version": bundle.get("version"),
            "dataset_size": bundle.get("dataset_size"),
            "best_algorithm": names.get(best[0], best[0]),
            "best_auc": round(best[1].get("auc", 0), 3),
            "best_accuracy": round(best[1].get("accuracy", 0), 3),
            "metrics": {names.get(k, k): v for k, v in metrics.items()},
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error al reentrenar: {e}"}), 500


@app.route("/api/admin/retrain_sse", methods=["POST"])
@admin_required
def api_admin_retrain_sse():
    """Igual que /api/admin/retrain pero responde SSE con progreso paso a paso."""
    from flask import Response, stream_with_context
    import queue, threading

    if "file" not in request.files:
        return jsonify({"error": "No se envio ningun archivo."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Archivo sin nombre."}), 400
    ext = f.filename.lower().rsplit(".", 1)[-1]
    raw = f.read()
    filename = f.filename

    # Parse the dataframe up front (before SSE stream starts)
    try:
        if ext == "csv":
            df = None
            for sep in [",", ";", "\t", "|"]:
                for enc in ["utf-8", "latin-1", "utf-8-sig"]:
                    try:
                        cand = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, engine="python")
                        if cand.shape[1] > 1:
                            df = cand
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
            if df is None:
                df = pd.read_csv(io.BytesIO(raw))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(raw))
        elif ext == "sav":
            import pyreadstat, tempfile
            with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
                tmp.write(raw); tmp_path = tmp.name
            df, _ = pyreadstat.read_sav(tmp_path)
            try: os.unlink(tmp_path)
            except Exception: pass
        else:
            return jsonify({"error": f"Formato no soportado: .{ext}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error al leer archivo: {e}"}), 500

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return jsonify({"error": "Faltan columnas: " + ", ".join(missing),
                        "required": REQUIRED_COLUMNS, "received": list(df.columns)}), 400

    q = queue.Queue()

    def progress_cb(step_id, step_idx, total, detail):
        q.put({"step": step_id, "idx": step_idx, "total": total, "detail": detail})

    def train_worker():
        try:
            from engine import train_and_persist
            bundle = train_and_persist(df=df, source=filename, on_progress=progress_cb)
            ENGINE.bundle = bundle
            q.put({"done": True, "bundle": {
                "version": bundle.get("version"),
                "dataset_size": bundle.get("dataset_size"),
                "metrics": bundle.get("metrics", {}),
                "approval_rate": bundle.get("approval_rate"),
                "class_distribution": bundle.get("class_distribution"),
                "n_features": bundle.get("n_features"),
            }})
        except Exception as e:
            q.put({"error": str(e)})

    t = threading.Thread(target=train_worker, daemon=True)
    t.start()

    def generate():
        import json as _json
        while True:
            try:
                msg = q.get(timeout=120)
            except Exception:
                yield "data: " + _json.dumps({"error": "Timeout"}) + "\n\n"
                break
            yield "data: " + _json.dumps(msg, default=str) + "\n\n"
            if "done" in msg or "error" in msg:
                break

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/admin/pipeline_steps")
@admin_required
def api_pipeline_steps():
    """Devuelve los pasos del pipeline para la UI de progreso."""
    return jsonify({"steps": PIPELINE_STEPS})


@app.route("/api/admin/retrain_seed", methods=["POST"])
@admin_required
def api_admin_retrain_seed():
    """Reentrena con el dataset semilla (demo) sin necesidad de subir archivo."""
    try:
        from engine import train_and_persist
        bundle = train_and_persist(source="seed")
        ENGINE.bundle = bundle
        return jsonify({"ok": True, "message": "Modelo demo regenerado.",
                        "version": bundle.get("version")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── API CONSUMIDOR ────────────────────────────────────────────────────────────
@app.route("/api/options")
def api_options():
    return jsonify({"ok": True, **ENGINE.options()})


@app.route("/api/score", methods=["POST"])
def api_score():
    try:
        payload = request.get_json(silent=True) or {}
        # El frontend puede adjuntar quién evalúa (opcional). Lo separamos del
        # payload para no pasárselo al motor de inferencia.
        guest = payload.pop("_user", None) if isinstance(payload, dict) else None
        # Prioridad: usuario autenticado en sesión > nombre de invitado del cliente.
        uid = session.get("uid")
        user = jdb.get_user(uid) if uid else guest
        result = ENGINE.score(payload)
        # Persistir la encuesta + resultado (nunca rompe el scoring).
        try:
            eval_id = jdb.save_evaluation(payload, result, user)
            if eval_id:
                result["eval_id"] = eval_id
        except Exception:
            pass
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"No se pudo evaluar el perfil: {e}"}), 500


# ─── REGISTRO SENCILLO DE USUARIOS (nombre / apodo) ─────────────────────────────
# Solo guarda el nombre que la persona elige en la app. NO es autenticación real
# ni toca los modelos: es un registro ligero para que el admin vea quién usa la app.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USERS_FILE = os.path.join(DATA_DIR, "usuarios.json")


def _load_users() -> List[dict]:
    try:
        with open(USERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_users(users: List[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


@app.route("/api/register", methods=["POST"])
def api_register():
    """Registra (o reconoce) a un usuario por su nombre/apodo. Público (SQL)."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()[:40]
    if not name:
        return jsonify({"error": "Escribe un nombre o apodo."}), 400
    res = jdb.register_user(name)
    return jsonify({"ok": True, "id": res["id"], "name": res["name"],
                    "returning": res["returning"]})


# ─── CUENTAS REALES (email + contraseña + foto) ─────────────────────────────────
def _login_session(user: dict, remember: bool = True) -> None:
    """Inicia sesión del usuario en la cookie firmada de Flask."""
    session["uid"] = user["id"]
    session.permanent = bool(remember)   # mantener sesión ~30 días


@app.route("/api/signup", methods=["POST"])
def api_signup():
    """Crea una cuenta real e inicia sesión."""
    d = request.get_json(silent=True) or {}
    res = jdb.create_account(d.get("name", ""), d.get("email", ""), d.get("password", ""))
    if not res["ok"]:
        return jsonify({"error": res["error"]}), 400
    user = res["user"]
    # Foto opcional al registrarse
    avatar = d.get("avatar")
    if avatar:
        up = jdb.update_profile(user["id"], avatar=avatar)
        if up["ok"]:
            user = up["user"]
    _login_session(user, d.get("remember", True))
    return jsonify({"ok": True, "user": user})


@app.route("/api/login", methods=["POST"])
def api_login():
    """Inicia sesión con email + contraseña."""
    d = request.get_json(silent=True) or {}
    user = jdb.authenticate(d.get("email", ""), d.get("password", ""))
    if not user:
        return jsonify({"error": "Email o contraseña incorrectos."}), 401
    _login_session(user, d.get("remember", True))
    return jsonify({"ok": True, "user": user})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Cierra la sesión del usuario (no toca la sesión de admin)."""
    session.pop("uid", None)
    return jsonify({"ok": True})


@app.route("/api/me")
def api_me():
    """Devuelve el usuario autenticado (para restaurar sesión al abrir la app)."""
    uid = session.get("uid")
    user = jdb.get_user(uid) if uid else None
    if not user:
        return jsonify({"ok": True, "user": None})
    return jsonify({"ok": True, "user": user})


@app.route("/api/profile", methods=["POST"])
def api_profile():
    """Actualiza nombre / foto / contraseña del usuario en sesión."""
    uid = session.get("uid")
    if not uid:
        return jsonify({"error": "Inicia sesión para editar tu perfil."}), 401
    d = request.get_json(silent=True) or {}
    res = jdb.update_profile(
        uid,
        name=d.get("name"),
        avatar=d.get("avatar"),
        password=d.get("password"),
    )
    if not res["ok"]:
        return jsonify({"error": res["error"]}), 400
    return jsonify({"ok": True, "user": res["user"]})


@app.route("/api/admin/users")
@admin_required
def api_admin_users():
    """Lista de usuarios registrados (solo admin)."""
    users = jdb.list_users()
    total_visits = sum(int(u.get("visits", 0)) for u in users)
    total_evals = sum(int(u.get("evaluations", 0)) for u in users)
    return jsonify({"ok": True, "count": len(users),
                    "total_visits": total_visits,
                    "total_evaluations": total_evals, "users": users})


# ─── API ADMIN · BASE DE DATOS (evaluaciones recopiladas) ───────────────────────
@app.route("/api/admin/db_stats")
@admin_required
def api_admin_db_stats():
    """Resumen de la base de datos de evaluaciones recopiladas."""
    return jsonify({"ok": True, **jdb.stats()})


@app.route("/api/admin/evaluations")
@admin_required
def api_admin_evaluations():
    """Últimas evaluaciones guardadas (encuesta + resultado)."""
    try:
        limit = min(int(request.args.get("limit", 100)), 500)
    except Exception:
        limit = 100
    rows = jdb.list_evaluations(limit=limit)
    return jsonify({"ok": True, "count": len(rows), "evaluations": rows})


@app.route("/api/admin/label", methods=["POST"])
@admin_required
def api_admin_label():
    """El admin marca el resultado REAL (0/1) de una evaluación → reentrenamiento
    con verdad de campo en vez de la predicción del modelo."""
    data = request.get_json(silent=True) or {}
    ok = jdb.set_real_label(data.get("id"), data.get("aprobado_real"))
    return (jsonify({"ok": True}) if ok
            else (jsonify({"error": "No se pudo etiquetar."}), 400))


@app.route("/api/admin/export_dataset")
@admin_required
def api_admin_export_dataset():
    """Descarga las evaluaciones recopiladas como CSV con el esquema exacto del
    motor (listo para reentrenar)."""
    try:
        from flask import Response
        df = jdb.export_dataframe()
        if df.empty:
            return jsonify({"error": "Aún no hay evaluaciones guardadas."}), 400
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        fname = _dt.datetime.now().strftime("jnus_dataset_%Y%m%d_%H%M.csv")
        return Response(csv, mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/retrain_from_db", methods=["POST"])
@admin_required
def api_admin_retrain_from_db():
    """Reentrena el modelo de producción con las evaluaciones recopiladas en la
    base de datos. Requiere un mínimo de filas y ambas clases presentes."""
    try:
        df = jdb.export_dataframe()
        if len(df) < 50:
            return jsonify({"error": f"Necesitas al menos 50 evaluaciones para reentrenar "
                                     f"(tienes {len(df)})."}), 400
        if df["aprobado"].nunique() < 2:
            return jsonify({"error": "Los datos recopilados tienen una sola clase. "
                                     "Se necesitan casos aprobados y rechazados."}), 400
        bundle = retrain_from_dataframe(df, source="base_de_datos")
        metrics = bundle.get("metrics", {})
        best = max(metrics.items(), key=lambda kv: kv[1].get("auc", 0)) if metrics else (None, {})
        names = {"logit": "Regresión Logística", "random_forest": "Random Forest",
                 "xgboost": "XGBoost", "neural_net": "Red Neuronal"}
        return jsonify({
            "ok": True,
            "message": f"Modelo reentrenado con {len(df)} evaluaciones recopiladas.",
            "version": bundle.get("version"),
            "dataset_size": bundle.get("dataset_size"),
            "best_algorithm": names.get(best[0], best[0]),
            "best_auc": round(best[1].get("auc", 0), 3),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error al reentrenar: {e}"}), 500


@app.route("/api/model_info")
def api_model_info():
    """Solo-lectura: metadatos del modelo activo (panel admin). NO entrena nada."""
    try:
        if not ENGINE.ready():
            ENGINE.bootstrap()
        b = ENGINE.bundle or {}
        metrics = b.get("metrics", {})
        # mejor algoritmo por AUC
        best, best_auc = None, 0.0
        for name, m in metrics.items():
            auc = float(m.get("auc", 0))
            if auc >= best_auc:
                best, best_auc = name, auc
        # fecha de entrenamiento = mtime del .pkl persistido
        import datetime as _dt
        train_date = None
        try:
            if os.path.exists(BUNDLE_PATH):
                train_date = _dt.datetime.fromtimestamp(
                    os.path.getmtime(BUNDLE_PATH)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        names = {"logit": "Regresión Logística", "random_forest": "Random Forest",
                 "xgboost": "XGBoost", "neural_net": "Red Neuronal"}
        best_acc = float(metrics.get(best, {}).get("accuracy", 0)) if best else 0.0
        return jsonify({
            "ok": True,
            "version": b.get("version", "1.0"),
            "training_date": train_date or "—",
            "dataset_size": b.get("dataset_size", 2500),
            "best_algorithm": names.get(best, best or "—"),
            "best_auc": round(best_auc, 3),
            "best_accuracy": round(best_acc, 3),
            "source": b.get("source", "seed"),
            "approval_rate": round(b.get("approval_rate", 0) * 100, 1),
            "class_distribution": b.get("class_distribution"),
            "n_features": b.get("n_features", len(b.get("columns", []))),
            "metrics": {names.get(k, k): v for k, v in metrics.items()},
            "bundle_path": os.path.basename(BUNDLE_PATH),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/learning")
@admin_required
def api_admin_learning():
    """Artefactos de aprendizaje del modelo activo (solo lectura, NO entrena):
    curva de pérdida de la red, coeficientes/signos del logit, importancia de
    variables y arquitectura. Para verificar que la red realmente aprende."""
    try:
        if not ENGINE.ready():
            ENGINE.bootstrap()
        b = ENGINE.bundle or {}
        learning = b.get("learning") or {}
        return jsonify({
            "ok": True,
            "learning": learning,
            "metrics": b.get("metrics", {}),
            "dataset_size": b.get("dataset_size"),
            "version": b.get("version"),
            "source": b.get("source"),
            "has_artifacts": bool(learning.get("nn_loss_curve")),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/db_schema")
@admin_required
def api_admin_db_schema():
    """Esquema + conteo de filas de la base SQLite (vista web para verificar)."""
    try:
        conn = jdb._connect()
        tables = []
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for r in rows:
            t = r["name"]
            cols = [{"name": c["name"], "type": c["type"]}
                    for c in conn.execute(f"PRAGMA table_info({t})").fetchall()]
            n = conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
            tables.append({"table": t, "rows": n, "columns": cols})
        conn.close()
        size = os.path.getsize(jdb.DB_PATH) if os.path.exists(jdb.DB_PATH) else 0
        return jsonify({"ok": True, "db_path": jdb.DB_PATH,
                        "size_kb": round(size / 1024, 1), "tables": tables})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/download_db")
@admin_required
def api_admin_download_db():
    """Descarga el archivo SQLite tal cual (para inspeccionar o respaldar)."""
    if not os.path.exists(jdb.DB_PATH):
        return jsonify({"error": "Aún no existe la base de datos."}), 404
    return send_file(jdb.DB_PATH, as_attachment=True,
                     download_name="janus.db",
                     mimetype="application/x-sqlite3")


@app.route("/manifest.webmanifest")
def manifest():
    return send_from_directory(app.static_folder, "manifest.webmanifest",
                               mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    # Servido desde la raíz para que el scope cubra /app
    resp = send_from_directory(app.static_folder, "sw.js", mimetype="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "xgboost": HAS_XGBOOST,
        "statsmodels": HAS_STATSMODELS,
        "version": "2.0.0",
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_state()
    return jsonify({"ok": True})


@app.route("/api/log")
def api_log():
    return jsonify({"log": STATE["log"][-200:]})


# ─── UPLOAD ────────────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        reset_state()
        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo (campo 'file' vacío)."}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Archivo sin nombre."}), 400

        ext = f.filename.lower().rsplit(".", 1)[-1]
        raw = f.read()
        log(f"📥 Recibido archivo '{f.filename}' ({len(raw):,} bytes)")

        # Guardar copia (opcional, para debugging)
        save_path = os.path.join(UPLOAD_DIR, f.filename)
        try:
            with open(save_path, "wb") as out:
                out.write(raw)
        except Exception:
            pass

        # Leer según extensión
        if ext == "csv":
            df = None
            for sep in [",", ";", "\t", "|"]:
                for enc in ["utf-8", "latin-1", "utf-8-sig"]:
                    try:
                        df_try = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, engine="python")
                        if df_try.shape[1] > 1:
                            df = df_try
                            log(f"✅ CSV leído (sep='{sep}', encoding='{enc}')")
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
            if df is None:
                # último intento: autodetect
                df = pd.read_csv(io.BytesIO(raw))
        elif ext in ("xlsx", "xls"):
            try:
                df = pd.read_excel(io.BytesIO(raw))
                log(f"✅ Excel leído")
            except ImportError:
                return jsonify({"error": "Falta 'openpyxl'. Instala: pip install openpyxl"}), 500
        else:
            return jsonify({"error": f"Formato no soportado: .{ext}. Usa CSV o Excel."}), 400

        if df.empty or df.shape[1] == 0:
            return jsonify({"error": "El archivo está vacío o no tiene columnas reconocibles."}), 400

        # Limpiar nombres de columnas
        df.columns = [str(c).strip() for c in df.columns]
        STATE["df_raw"] = df

        numeric, categorical = detect_column_types(df)
        STATE["numeric_cols"] = numeric
        STATE["categorical_cols"] = categorical

        log(f"📊 Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        log(f"🔢 Numéricas detectadas ({len(numeric)}): {', '.join(numeric[:6])}{'…' if len(numeric) > 6 else ''}")
        log(f"🔤 Categóricas detectadas ({len(categorical)}): {', '.join(categorical[:6])}{'…' if len(categorical) > 6 else ''}")

        preview = df.head(8).fillna("").astype(object).where(pd.notnull(df.head(8)), None)
        preview_records = []
        for _, row in preview.iterrows():
            preview_records.append({str(k): safe_jsonable(v) for k, v in row.items()})

        # Sugerir target binario
        suggested_target = None
        for c in df.columns:
            s = df[c].dropna()
            uniq = s.unique()
            if len(uniq) == 2:
                suggested_target = c
                break

        return jsonify({
            "ok": True,
            "filename": f.filename,
            "rows": int(df.shape[0]),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(df.columns),
            "numeric": numeric,
            "categorical": categorical,
            "preview": preview_records,
            "missing_total": int(df.isna().sum().sum()),
            "suggested_target": suggested_target,
            "log": STATE["log"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error al leer archivo: {e}"}), 500


# ─── PREPROCESS ────────────────────────────────────────────────────────────────
@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    try:
        if STATE["df_raw"] is None:
            return jsonify({"error": "Sube un archivo primero."}), 400

        body = request.get_json(silent=True) or {}
        target = body.get("target")
        if not target:
            return jsonify({"error": "Selecciona la variable objetivo (target)."}), 400

        df = STATE["df_raw"].copy()
        if target not in df.columns:
            return jsonify({"error": f"La columna '{target}' no existe en el dataset."}), 400

        steps: List[str] = []
        STATE["log"] = []  # reinicia log de esta fase
        log(f"🎯 Variable objetivo: {target}")

        # 1) Eliminar filas con target NaN
        before = len(df)
        df = df.dropna(subset=[target])
        removed = before - len(df)
        if removed:
            steps.append(f"Eliminadas {removed} filas con target nulo.")
            log(f"🗑️ {removed} filas con target nulo eliminadas")

        # 2) Normalizar target a 0/1
        df[target] = normalize_binary_text(df[target])
        try:
            df[target] = pd.to_numeric(df[target], errors="raise").astype(int)
        except Exception:
            # Si el target no es binario numérico, mapear sus 2 valores únicos
            uniq = df[target].dropna().unique()
            if len(uniq) != 2:
                return jsonify({
                    "error": f"El target '{target}' debe ser binario (2 clases). Encontradas: {len(uniq)}"
                }), 400
            mapping = {uniq[0]: 0, uniq[1]: 1}
            df[target] = df[target].map(mapping).astype(int)
            steps.append(f"Target binarizado: {uniq[0]}→0, {uniq[1]}→1")
            log(f"🔁 Target binarizado: {uniq[0]}→0, {uniq[1]}→1")

        y = df[target].astype(int)
        X = df.drop(columns=[target])

        # 3) Convertir booleanos textuales
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = normalize_binary_text(X[col])

        # 4) Detectar tipos en X
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        # 5) Imputar numéricas con mediana
        median_map = {}
        for c in numeric_cols:
            med = X[c].median()
            if pd.isna(med):
                med = 0.0
            n_na = int(X[c].isna().sum())
            if n_na:
                X[c] = X[c].fillna(med)
                steps.append(f"Imputados {n_na} valores nulos en '{c}' con mediana={med:.3f}")
            median_map[c] = float(med)

        # 6) Imputar categóricas con moda
        mode_map = {}
        for c in categorical_cols:
            n_na = int(X[c].isna().sum())
            mode = X[c].mode()
            mode_val = str(mode.iloc[0]) if len(mode) else "MISSING"
            if n_na:
                X[c] = X[c].fillna(mode_val)
                steps.append(f"Imputados {n_na} nulos en '{c}' con moda='{mode_val}'")
            mode_map[c] = mode_val

        # 7) One-Hot encoding de categóricas (drop_first para evitar dummy-trap)
        if categorical_cols:
            before_cols = len(X.columns)
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
            steps.append(f"One-Hot encoding: {len(categorical_cols)} columnas → {len(X.columns) - before_cols + len(categorical_cols)} dummies")
            log(f"🧬 One-Hot: {categorical_cols} → {len(X.columns)} columnas totales")

        # 8) Eliminar columnas constantes
        const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if const_cols:
            X = X.drop(columns=const_cols)
            steps.append(f"Eliminadas {len(const_cols)} columnas constantes.")

        # 9) Cast a float
        X = X.astype(float)

        # 10) Train/test split
        test_size = float(body.get("test_size", 0.2))
        test_size = max(0.1, min(test_size, 0.5))

        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        # 11) Scaler
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        STATE["df_clean"] = df
        STATE["X_train"] = X_train_sc
        STATE["X_test"] = X_test_sc
        STATE["X_train_raw"] = X_train  # sin escalar (para statsmodels)
        STATE["X_test_raw"] = X_test
        STATE["y_train"] = y_train.values
        STATE["y_test"] = y_test.values
        STATE["scaler"] = scaler
        STATE["feature_names"] = list(X.columns)
        STATE["target"] = target
        STATE["median_map"] = median_map
        STATE["mode_map"] = mode_map
        # Rangos de cada feature (para los inputs del form de predicción)
        ranges = {}
        for c in X.columns:
            ranges[c] = {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
                "mean": float(X[c].mean()),
            }
        STATE["feature_ranges"] = ranges

        class_counts = y.value_counts().to_dict()
        steps.append(f"Split: {len(X_train)} train · {len(X_test)} test (test_size={int(test_size*100)}%)")
        steps.append(f"Estandarización: StandardScaler aplicado")

        log(f"✅ Preprocesamiento completo · {len(X.columns)} features")

        return jsonify({
            "ok": True,
            "steps": steps,
            "n_features": int(len(X.columns)),
            "feature_names": list(X.columns)[:50],
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "class_dist": {
                "0": int(class_counts.get(0, 0)),
                "1": int(class_counts.get(1, 0)),
            },
            "target": target,
            "log": STATE["log"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error al preprocesar: {e}"}), 500


# ─── TRAIN ─────────────────────────────────────────────────────────────────────
def _build_model(name: str):
    if name == "logit":
        return LogisticRegression(max_iter=1000, solver="lbfgs"), True
    if name == "probit":
        if HAS_STATSMODELS:
            return "PROBIT_SM", True  # marker
        # fallback: LogisticRegression
        return LogisticRegression(max_iter=1000, solver="lbfgs"), True
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1), False
    if name == "xgboost":
        if HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0,
            ), False
        return RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1), False
    if name == "neural_net":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                             random_state=42, early_stopping=True), True
    raise ValueError(f"Modelo desconocido: {name}")


@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        if STATE["X_train"] is None:
            return jsonify({"error": "Ejecuta el preprocesamiento primero."}), 400

        body = request.get_json(silent=True) or {}
        model_name = body.get("model", "logit")

        STATE["log"] = []
        log(f"🤖 Iniciando entrenamiento: {model_name}")

        model, use_scaled = _build_model(model_name)

        Xtr = STATE["X_train"] if use_scaled else STATE["X_train_raw"].values
        Xte = STATE["X_test"] if use_scaled else STATE["X_test_raw"].values
        ytr = STATE["y_train"]
        yte = STATE["y_test"]

        feature_importance = []

        # ── Probit con statsmodels ──
        sm_model = None
        if model == "PROBIT_SM" and HAS_STATSMODELS:
            log("📐 Ajustando Probit con statsmodels (MLE)")
            X_sm = sm.add_constant(Xtr, has_constant="add")
            sm_model = sm.Probit(ytr, X_sm).fit(disp=False, maxiter=200)
            X_te_sm = sm.add_constant(Xte, has_constant="add")
            y_prob = sm_model.predict(X_te_sm)
            y_pred = (y_prob >= 0.5).astype(int)
            coefs = sm_model.params[1:]  # quitar intercepto
            for name, c in zip(STATE["feature_names"], coefs):
                feature_importance.append({"name": name, "value": float(abs(c))})
        else:
            log(f"🏋️ Entrenando {model.__class__.__name__}")
            model.fit(Xtr, ytr)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(Xte)[:, 1]
            else:
                y_prob = model.decision_function(Xte)
                y_prob = 1 / (1 + np.exp(-y_prob))
            y_pred = (y_prob >= 0.5).astype(int)

            # importancia
            if hasattr(model, "feature_importances_"):
                for name, v in zip(STATE["feature_names"], model.feature_importances_):
                    feature_importance.append({"name": name, "value": float(v)})
            elif hasattr(model, "coef_"):
                coefs = model.coef_[0]
                for name, c in zip(STATE["feature_names"], coefs):
                    feature_importance.append({"name": name, "value": float(abs(c))})

        feature_importance.sort(key=lambda d: d["value"], reverse=True)

        # ── Métricas ──
        acc = float(accuracy_score(yte, y_pred))
        try:
            auc = float(roc_auc_score(yte, y_prob))
        except Exception:
            auc = 0.5
        gini = 2 * auc - 1
        prec = float(precision_score(yte, y_pred, zero_division=0))
        rec = float(recall_score(yte, y_pred, zero_division=0))
        f1 = float(f1_score(yte, y_pred, zero_division=0))
        cm = confusion_matrix(yte, y_pred, labels=[0, 1]).tolist()

        log(f"📈 Accuracy={acc:.3f} · AUC={auc:.3f} · Gini={gini:.3f}")

        # ── Juan: cliente medio (mediana de features no escaladas) ──
        X_train_raw = STATE["X_train_raw"]
        juan_raw = X_train_raw.median().values.reshape(1, -1)
        if use_scaled:
            juan_input = STATE["scaler"].transform(juan_raw)
        else:
            juan_input = juan_raw

        if model == "PROBIT_SM" and HAS_STATSMODELS:
            j_in = sm.add_constant(juan_input, has_constant="add")
            juan_p = float(sm_model.predict(j_in)[0])
        elif hasattr(model, "predict_proba"):
            juan_p = float(model.predict_proba(juan_input)[0, 1])
        else:
            juan_p = float(1 / (1 + np.exp(-model.decision_function(juan_input)[0])))

        log(f"👤 Juan (cliente medio) · P(pago) = {juan_p*100:.1f}%")

        # Persistir modelo para /api/predict y visualizaciones reales
        STATE["model"] = model if model != "PROBIT_SM" else None
        STATE["sm_model"] = sm_model
        STATE["model_name"] = model_name
        STATE["use_scaled"] = use_scaled

        # Sample para gráficos (max 500 puntos)
        n_sample = min(500, len(y_prob))
        idx = np.random.RandomState(42).choice(len(y_prob), n_sample, replace=False) if len(y_prob) > n_sample else np.arange(len(y_prob))

        # ── ROC curve ──
        roc_points = []
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(yte, y_prob)
            # samplea 100 puntos
            step = max(1, len(fpr) // 100)
            for i in range(0, len(fpr), step):
                roc_points.append({"fpr": float(fpr[i]), "tpr": float(tpr[i])})
            roc_points.append({"fpr": 1.0, "tpr": 1.0})
        except Exception:
            pass

        result = {
            "ok": True,
            "model": model_name,
            "accuracy": acc,
            "auc": auc,
            "gini": gini,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "train_size": int(len(ytr)),
            "test_size": int(len(yte)),
            "confusion_matrix": cm,
            "feature_importance": feature_importance[:15],
            "juan_probability": round(juan_p * 100, 2),
            "juan_verdict_threshold": 0.5,
            "y_prob_sample": [float(y_prob[i]) for i in idx],
            "y_test_sample": [int(yte[i]) for i in idx],
            "roc_points": roc_points,
            "layers": [len(STATE["feature_names"]), 64, 32, 1],
            "tree_depth": 4,
            "log": STATE["log"],
        }
        return jsonify(safe_jsonable(result))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error en entrenamiento: {e}"}), 500


# ─── PREDICT (formulario en tiempo real) ───────────────────────────────────────
@app.route("/api/feature_form", methods=["GET"])
def api_feature_form():
    """Devuelve el esquema del formulario: features, rangos, mediana."""
    if STATE["feature_names"] is None:
        return jsonify({"error": "Preprocesa primero."}), 400
    ranges = STATE.get("feature_ranges") or {}
    fields = []
    for f in STATE["feature_names"]:
        r = ranges.get(f, {"min": 0, "max": 1, "median": 0.5, "mean": 0.5})
        is_binary = bool(set([r["min"], r["max"]]).issubset({0.0, 1.0}))
        fields.append({
            "name": f,
            "min": r["min"],
            "max": r["max"],
            "median": r["median"],
            "mean": r["mean"],
            "binary": is_binary,
        })
    return jsonify({"ok": True, "fields": fields, "target": STATE.get("target")})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predicción en tiempo real con un vector custom + contribuciones por feature."""
    try:
        if STATE.get("model") is None and STATE.get("sm_model") is None:
            return jsonify({"error": "Entrena un modelo primero."}), 400

        body = request.get_json(silent=True) or {}
        values = body.get("values", {}) or {}
        feats = STATE["feature_names"]
        ranges = STATE.get("feature_ranges") or {}

        # Construir vector en el orden correcto
        row = []
        used_values = {}
        for f in feats:
            if f in values and values[f] is not None and values[f] != "":
                try:
                    v = float(values[f])
                except Exception:
                    v = ranges.get(f, {}).get("median", 0.0)
            else:
                v = ranges.get(f, {}).get("median", 0.0)
            row.append(v)
            used_values[f] = v

        x_in = _row_to_input(row)
        prob = _compute_prob(x_in)

        # Contribuciones por feature
        contribs = []
        m = STATE.get("model")
        sm_m = STATE.get("sm_model")
        if sm_m is not None:
            coefs = sm_m.params[1:]
            for name, c, v in zip(feats, coefs, x_in[0]):
                contribs.append({
                    "name": name, "value": float(v),
                    "raw": float(used_values[name]),
                    "contribution": float(c * v),
                })
        elif m is not None and hasattr(m, "coef_"):
            for name, c, v in zip(feats, m.coef_[0], x_in[0]):
                contribs.append({
                    "name": name, "value": float(v),
                    "raw": float(used_values[name]),
                    "contribution": float(c * v),
                })
        elif m is not None and hasattr(m, "feature_importances_"):
            for name, imp, v in zip(feats, m.feature_importances_, x_in[0]):
                contribs.append({
                    "name": name, "value": float(v),
                    "raw": float(used_values[name]),
                    "contribution": float(imp * v),
                })
        else:
            # Modelos sin coeficientes (red neuronal MLP): atribución local por
            # ABLACIÓN — cuánto cae la probabilidad si esta variable vuelve a su
            # mediana. Da un XAI real para la red neuronal.
            ranges_local = STATE.get("feature_ranges") or {}
            for i, name in enumerate(feats):
                ablated = list(row)
                ablated[i] = ranges_local.get(name, {}).get("median", 0.0)
                p_ab = _compute_prob(_row_to_input(ablated))
                contribs.append({
                    "name": name, "value": float(x_in[0][i]),
                    "raw": float(used_values[name]),
                    "contribution": float(prob - p_ab),
                })

        contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)

        # ── XAI · análisis what-if (educación financiera) ──────────────────────
        # Para las top features, probamos pequeñas variaciones y medimos cuánto
        # sube/baja la probabilidad. Funciona para CUALQUIER modelo (lineal o árbol)
        # porque mide la sensibilidad real del modelo entrenado.
        base_row = [used_values[f] for f in feats]
        sensitivity = []
        for c in contribs[:12]:
            fname = c["name"]
            i = feats.index(fname)
            r = ranges.get(fname, {"min": 0.0, "max": 1.0})
            fmin, fmax = float(r.get("min", 0.0)), float(r.get("max", 1.0))
            cur = base_row[i]
            binary = set([fmin, fmax]).issubset({0.0, 1.0})
            candidates = ([1.0 - cur] if binary
                          else [v for v in (min(fmax, cur + (fmax - fmin) * 0.1),
                                            max(fmin, cur - (fmax - fmin) * 0.1)) if v != cur])
            best = None
            for cand in candidates:
                row2 = list(base_row); row2[i] = cand
                p2 = _compute_prob(_row_to_input(row2))
                if best is None or (p2 - prob) > best["delta"]:
                    best = {"value": float(cand), "delta": float(p2 - prob), "prob": float(p2)}
            if best is not None:
                sensitivity.append({
                    "name": fname, "current": float(cur), "binary": bool(binary),
                    "suggested": best["value"], "delta": best["delta"], "new_prob": best["prob"],
                })

        improvers = sorted([s for s in sensitivity if s["delta"] > 0.005],
                           key=lambda s: s["delta"], reverse=True)[:4]
        tips = []
        for s in improvers:
            up = s["suggested"] > s["current"]
            if s["binary"]:
                action = "Activa" if s["suggested"] >= 0.5 else "Desactiva"
                tips.append(f"{action} «{s['name']}» → tu probabilidad subiría +{s['delta']*100:.1f} pts (a {s['new_prob']*100:.0f}%).")
            else:
                verb = "Aumentar" if up else "Reducir"
                tips.append(f"{verb} «{s['name']}» mejoraría tu aprobación en +{s['delta']*100:.1f} pts (a {s['new_prob']*100:.0f}%).")
        if not tips:
            tips.append("Tu perfil ya está bien optimizado para este modelo: ninguna variable individual mejora mucho la probabilidad.")

        positive = [c for c in contribs if c["contribution"] > 0][:6]
        negative = [c for c in contribs if c["contribution"] < 0][:6]

        decision = "APROBAR" if prob > 0.65 else ("REVISAR" if prob > 0.4 else "RECHAZAR")
        explanation = (
            f"El modelo ({STATE.get('model_name','?')}) estima una probabilidad de pago del {prob*100:.1f}%. "
            f"Las 3 variables con mayor peso en esta decisión son: "
            + ", ".join([f"{c['name']} (Δ={c['contribution']:+.3f})" for c in contribs[:3]])
            + "."
        )

        return jsonify(safe_jsonable({
            "ok": True,
            "probability": prob,
            "decision": decision,
            "threshold": 0.5,
            "model": STATE.get("model_name"),
            "contributions": contribs[:20],
            "explanation": explanation,
            "xai": {
                "positive": positive,
                "negative": negative,
                "improvers": improvers,
                "tips": tips,
            },
        }))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── TREE STRUCTURE (real, sacada del modelo) ─────────────────────────────────
def _extract_tree(tree, feature_names, max_depth=4):
    """Convierte un sklearn tree_ en dict recursivo para D3/canvas."""
    t = tree.tree_

    def build(node_id, depth):
        if node_id == -1 or depth > max_depth:
            return None
        is_leaf = t.children_left[node_id] == t.children_right[node_id]
        value = t.value[node_id][0]
        total = float(value.sum())
        n0, n1 = float(value[0]) if len(value) > 0 else 0.0, float(value[1]) if len(value) > 1 else 0.0
        node = {
            "id": int(node_id),
            "depth": depth,
            "samples": int(t.n_node_samples[node_id]),
            "value": [n0, n1],
            "proba": (n1 / total) if total > 0 else 0.0,
            "is_leaf": bool(is_leaf or depth == max_depth),
        }
        if not node["is_leaf"]:
            fi = int(t.feature[node_id])
            node["feature"] = feature_names[fi] if 0 <= fi < len(feature_names) else f"x{fi}"
            node["threshold"] = float(t.threshold[node_id])
            node["left"] = build(int(t.children_left[node_id]), depth + 1)
            node["right"] = build(int(t.children_right[node_id]), depth + 1)
        return node

    return build(0, 0)


@app.route("/api/tree_data", methods=["GET"])
def api_tree_data():
    """Devuelve la estructura real de un árbol (RF/XGB) para la visualización."""
    try:
        m = STATE.get("model")
        if m is None:
            return jsonify({"error": "Entrena un modelo de árbol primero (RF o XGBoost)."}), 400

        feats = STATE["feature_names"]

        if hasattr(m, "estimators_"):
            # Random Forest — varios árboles
            trees = []
            for est in m.estimators_[:6]:
                trees.append(_extract_tree(est, feats, max_depth=4))
            return jsonify(safe_jsonable({
                "ok": True,
                "kind": "forest",
                "n_estimators": len(m.estimators_),
                "trees": trees,
            }))
        elif HAS_XGBOOST and hasattr(m, "get_booster"):
            # XGBoost — usar dump_model
            booster = m.get_booster()
            dump = booster.get_dump(dump_format="json")[:6]
            import json as _json
            trees = [_json.loads(t) for t in dump]
            return jsonify(safe_jsonable({
                "ok": True,
                "kind": "xgboost",
                "n_estimators": len(booster.get_dump()),
                "trees_raw": trees,
            }))
        else:
            return jsonify({"error": f"El modelo {STATE.get('model_name')} no tiene árboles."}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── NEURAL NET ACTIVATIONS ───────────────────────────────────────────────────
@app.route("/api/nn_activations", methods=["POST"])
def api_nn_activations():
    """Forward-pass de un vector por la red MLP. Devuelve activaciones por capa."""
    try:
        m = STATE.get("model")
        if m is None or not hasattr(m, "coefs_"):
            return jsonify({"error": "Entrena la Red Neuronal primero."}), 400

        body = request.get_json(silent=True) or {}
        values = body.get("values")
        feats = STATE["feature_names"]
        ranges = STATE.get("feature_ranges") or {}

        if values is None:
            x = np.array([ranges.get(f, {}).get("median", 0.0) for f in feats]).reshape(1, -1)
        else:
            x = np.array([
                float(values.get(f, ranges.get(f, {}).get("median", 0.0)))
                for f in feats
            ]).reshape(1, -1)

        if STATE.get("use_scaled") and STATE.get("scaler") is not None:
            x = STATE["scaler"].transform(x)

        # Forward pass manual usando coefs_ / intercepts_
        a = x[0]
        layers = [{"name": "Input", "activations": a.tolist()}]
        for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
            z = a @ W + b
            # ReLU en ocultas, sigmoid/softmax en salida
            if i < len(m.coefs_) - 1:
                a = np.maximum(0, z)
                layers.append({"name": f"Hidden {i+1} ({len(b)})", "activations": a.tolist()})
            else:
                a = 1 / (1 + np.exp(-z))
                layers.append({"name": "Output σ", "activations": a.tolist()})

        return jsonify(safe_jsonable({
            "ok": True,
            "layers": layers,
            "prediction": float(a[-1]) if len(a) else 0.0,
            "weights_shapes": [list(W.shape) for W in m.coefs_],
        }))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# MAIN  (dev local: `python app.py`  ·  prod: gunicorn app:app)
# ──────────────────────────────────────────────────────────────────────────────
# Pre-cargar / entrenar los modelos del producto al iniciar (una sola vez).
# En la nube (gunicorn) corre al importar el módulo en cada worker.
if os.environ.get("JNUS_NO_WARM", os.environ.get("JANUS_NO_WARM")) != "1":
    try:
        ENGINE.bootstrap()
        print(f"[JNUS] Motor de inferencia listo · modelos: {list(ENGINE.bundle['models'].keys())}")
    except Exception as _e:
        print(f"[JNUS] Aviso: bootstrap diferido ({_e})")

# Migración única: usuarios del JSON antiguo → SQLite (si la tabla está vacía).
try:
    if not jdb.list_users():
        for u in _load_users():
            try:
                jdb.register_user(u.get("name", ""))
            except Exception:
                pass
except Exception:
    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    # Local/escritorio: sin debugger ni reloader (evita doble arranque y doble
    # apertura del navegador). El reloader era la razón del FLASK_ENV=production.
    debug = (not IS_LOCAL) and (os.environ.get("FLASK_ENV", "development") != "production")
    open_admin = os.environ.get("JNUS_OPEN_ADMIN") == "1"
    admin_url = f"http://127.0.0.1:{port}/admin/login"
    app_url = f"http://127.0.0.1:{port}/app"
    print("\n" + "=" * 60)
    print("  SIAC · JNUS AI — Credit Intelligence Backend")
    print("=" * 60)
    print(f"  ADMIN:   {admin_url}   (usuario: admin)")
    print(f"  App:     {app_url}")
    print(f"  XGBoost:     {'OK' if HAS_XGBOOST else 'no instalado (fallback RF)'}")
    print(f"  statsmodels: {'OK' if HAS_STATSMODELS else 'no instalado (fallback Logit)'}")
    print("=" * 60 + "\n")

    # Apertura automática del panel admin (SOLO local, si el launcher lo pide).
    # Producción/Render usa gunicorn y nunca entra aquí, así que no le afecta.
    if open_admin:
        import threading
        import urllib.request
        import webbrowser

        def _open_admin_when_ready():
            health = f"http://127.0.0.1:{port}/api/health"
            for _ in range(60):
                try:
                    urllib.request.urlopen(health, timeout=1)
                    webbrowser.open(admin_url)
                    return
                except Exception:
                    time.sleep(0.7)

        threading.Thread(target=_open_admin_when_ready, daemon=True).start()

    app.run(host=host, port=port, debug=debug, use_reloader=debug)
