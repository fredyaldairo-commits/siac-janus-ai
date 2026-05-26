"""
JNUS Credit AI Lab — Backend Flask
====================================
API REST para análisis de riesgo crediticio con IA + econometría.

Endpoints:
  GET  /                  → sirve el frontend (templates/index.html)
  GET  /api/health        → ping de salud
  POST /api/upload        → carga CSV/Excel y devuelve metadatos
  POST /api/preprocess    → limpieza + codificación de variables
  POST /api/train         → entrena el modelo seleccionado y devuelve métricas
  POST /api/reset         → limpia el estado de la sesión

Modelos soportados: logit, probit, random_forest, xgboost, neural_net
"""
from __future__ import annotations

import io
import os
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

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


APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB

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
# RUTAS
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


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

        x_raw = np.array(row).reshape(1, -1)
        use_scaled = STATE.get("use_scaled", True)
        x_in = STATE["scaler"].transform(x_raw) if use_scaled and STATE.get("scaler") is not None else x_raw

        # Predicción
        if STATE.get("sm_model") is not None:
            x_sm = sm.add_constant(x_in, has_constant="add")
            prob = float(STATE["sm_model"].predict(x_sm)[0])
        else:
            m = STATE["model"]
            if hasattr(m, "predict_proba"):
                prob = float(m.predict_proba(x_in)[0, 1])
            else:
                z = float(m.decision_function(x_in)[0])
                prob = 1.0 / (1.0 + float(np.exp(-z)))

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
            for name, v in zip(feats, x_in[0]):
                contribs.append({
                    "name": name, "value": float(v),
                    "raw": float(used_values[name]),
                    "contribution": 0.0,
                })

        contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)

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
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_ENV", "development") != "production"
    print("\n" + "=" * 60)
    print("  SIAC · JANUS AI — Credit Intelligence Backend")
    print("=" * 60)
    print(f"  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://{host}:{port}")
    print(f"  XGBoost:     {'OK' if HAS_XGBOOST else 'no instalado (fallback RF)'}")
    print(f"  statsmodels: {'OK' if HAS_STATSMODELS else 'no instalado (fallback Logit)'}")
    print("=" * 60 + "\n")
    app.run(host=host, port=port, debug=debug)
