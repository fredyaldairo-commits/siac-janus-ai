"""
JNUS AI · Consumer Inference Engine
=====================================
Capa de PRODUCTO (no toca el pipeline de ciencia de datos existente).

Responsabilidades:
  1. Generar un dataset semilla realista de crédito (contexto Ecuador) cuyas
     columnas SON exactamente los campos del formulario del usuario final.
  2. Entrenar UNA sola vez los 4 modelos (Logistic Regression, Random Forest,
     XGBoost, Neural Network) y persistirlos a disco (models/*.pkl).
  3. En arranque: si los .pkl existen → cargar; si no → entrenar y guardar.
  4. score(): inferencia instantánea (el usuario NUNCA entrena) + XAI en
     español plano (factores positivos / negativos / recomendaciones).

Los modelos y sus hiperparámetros son los MISMOS que usa el backend original.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import joblib
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
BUNDLE_PATH = os.path.join(MODELS_DIR, "janus_bundle.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# ESQUEMA DEL FORMULARIO  (= columnas del dataset semilla)
# ──────────────────────────────────────────────────────────────────────────────
CREDIT_TYPES = [
    "Hipotecario", "Vehicular", "Personal", "Microcrédito",
    "Productivo", "Emprendimiento",
]
EMPLOYMENT = [
    "Empleado Público", "Empleado Privado", "Emprendedor",
    "Negocio Propio", "Trabajo Informal", "Desempleado",
]
EDUCATION = ["Primaria", "Secundaria", "Universitaria", "Posgrado"]
PAYMENT_HISTORY = ["Malo", "Regular", "Bueno", "Excelente"]
INSTITUTIONS = [
    "Banco Pichincha", "Banco Guayaquil", "Produbanco",
    "Banco del Pacífico", "Cooperativa JEP",
]
SEX = ["Masculino", "Femenino"]

# Etiquetas amigables para mostrar en el XAI
FRIENDLY = {
    "edad": "Tu edad",
    "ingresos_mensuales": "Tus ingresos mensuales",
    "cargas_familiares": "Tus cargas familiares",
    "creditos_activos": "Tus créditos activos",
    "sexo": "Sexo",
    "educacion": "Tu nivel educativo",
    "historial_pagos": "Tu historial de pagos",
    "institucion": "La institución financiera",
    "tipo_credito": "El tipo de crédito",
    "situacion_laboral": "Tu situación laboral",
}


def _friendly(feature: str) -> str:
    """Convierte 'historial_pagos_Excelente' → 'Historial de pagos: Excelente'."""
    for base, label in FRIENDLY.items():
        if feature == base:
            return label
        if feature.startswith(base + "_"):
            val = feature[len(base) + 1:].replace("_", " ")
            return f"{label.replace('Tu ', '').replace('Tus ', '').capitalize()}: {val}"
    return feature.replace("_", " ").capitalize()


# ──────────────────────────────────────────────────────────────────────────────
# 1) DATASET SEMILLA SINTÉTICO (realista)
# ──────────────────────────────────────────────────────────────────────────────
def generate_seed(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        edad = int(rng.integers(20, 70))
        sexo = rng.choice(SEX)
        educacion = rng.choice(EDUCATION, p=[0.2, 0.4, 0.32, 0.08])
        cargas = int(rng.integers(0, 6))
        situacion = rng.choice(EMPLOYMENT, p=[0.15, 0.30, 0.15, 0.15, 0.18, 0.07])
        # ingresos correlacionados con situación y educación
        base_income = {
            "Empleado Público": 1100, "Empleado Privado": 1000, "Emprendedor": 900,
            "Negocio Propio": 1200, "Trabajo Informal": 550, "Desempleado": 250,
        }[situacion]
        edu_mult = {"Primaria": 0.8, "Secundaria": 1.0, "Universitaria": 1.45, "Posgrado": 2.0}[educacion]
        ingresos = max(0, float(rng.normal(base_income * edu_mult, 350)))
        creditos = int(rng.integers(0, 6))
        historial = rng.choice(PAYMENT_HISTORY, p=[0.15, 0.25, 0.35, 0.25])
        institucion = rng.choice(INSTITUTIONS)
        tipo = rng.choice(CREDIT_TYPES)

        # ── función latente de aprobación ──
        z = -1.2
        z += (ingresos / 1500.0) * 1.6
        z += {"Malo": -1.6, "Regular": -0.4, "Bueno": 0.7, "Excelente": 1.7}[historial]
        z += {"Primaria": -0.3, "Secundaria": 0.0, "Universitaria": 0.5, "Posgrado": 0.9}[educacion]
        z += -0.45 * creditos
        z += -0.15 * cargas
        z += {"Empleado Público": 0.8, "Empleado Privado": 0.6, "Emprendedor": 0.1,
              "Negocio Propio": 0.4, "Trabajo Informal": -0.6, "Desempleado": -1.8}[situacion]
        z += {"Microcrédito": 0.5, "Personal": 0.2, "Emprendimiento": 0.0,
              "Productivo": -0.1, "Vehicular": -0.2, "Hipotecario": -0.5}[tipo]
        z += -0.012 * abs(edad - 40)  # edad media favorece
        z += float(rng.normal(0, 0.6))  # ruido

        prob = 1.0 / (1.0 + np.exp(-z))
        aprobado = int(rng.random() < prob)
        rows.append({
            "edad": edad, "sexo": sexo, "educacion": educacion,
            "cargas_familiares": cargas, "ingresos_mensuales": round(ingresos, 2),
            "creditos_activos": creditos, "historial_pagos": historial,
            "institucion": institucion, "tipo_credito": tipo,
            "situacion_laboral": situacion, "aprobado": aprobado,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 2) ENTRENAMIENTO + PERSISTENCIA
# ──────────────────────────────────────────────────────────────────────────────
RAW_NUM = ["edad", "ingresos_mensuales", "cargas_familiares", "creditos_activos"]
RAW_CAT = ["sexo", "educacion", "historial_pagos", "institucion",
           "tipo_credito", "situacion_laboral"]
TARGET = "aprobado"


def _encode(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    X = pd.get_dummies(df[RAW_NUM + RAW_CAT], columns=RAW_CAT, drop_first=False, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


class GeluMLP:
    """Red neuronal feed-forward PROPIA de JNUS (NumPy puro, sin TensorFlow).

    Estructura probabilística pedida:
      - Capas OCULTAS con activación GELU  →  g(z)=0.5·z·(1+tanh(√(2/π)(z+0.044715 z³)))
        GELU es suave (derivable en todo punto): acelera el descenso por
        gradiente y, a diferencia de ReLU, NO mata neuronas a 0.
      - Capa de SALIDA con SIGMOIDE  →  σ(z)=1/(1+e^{-z})  →  probabilidad ∈ (0,1)
        que evita que la decisión colapse a 0/1 duro (mantiene el matiz).
      - Pérdida = entropía cruzada binaria (BCE), optimizada con backprop + Adam.

    Expone atributos al estilo sklearn (coefs_, intercepts_, loss_curve_) para
    reutilizar la extracción de artefactos del panel admin.
    """

    _C = 0.7978845608028654  # √(2/π)

    def __init__(self, hidden=(64, 32), epochs=150, lr=0.01, l2=5e-4,
                 batch=64, patience=12, seed=42):
        self.hidden = tuple(hidden)
        self.epochs = int(epochs)          # máximo (corta antes con early stopping)
        self.lr = float(lr)
        self.l2 = float(l2)                 # regularización L2 (evita memorizar)
        self.batch = int(batch)
        self.patience = int(patience)       # paciencia de early stopping
        self.seed = int(seed)
        self.classes_ = np.array([0, 1])

    # ── activaciones ──
    @classmethod
    def _gelu(cls, x):
        return 0.5 * x * (1.0 + np.tanh(cls._C * (x + 0.044715 * np.power(x, 3))))

    @classmethod
    def _gelu_grad(cls, x):
        u = cls._C * (x + 0.044715 * np.power(x, 3))
        t = np.tanh(u)
        du = cls._C * (1.0 + 3 * 0.044715 * np.square(x))
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _init_params(self, n_in):
        rng = np.random.default_rng(self.seed)
        sizes = [n_in] + list(self.hidden) + [1]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])           # init tipo He (apto GELU/ReLU)
            self.W.append(rng.normal(0.0, scale, size=(sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def _forward(self, X, cache=False):
        a = X
        zs, acts = [], []
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            a = self._gelu(z) if i < L - 1 else self._sigmoid(z)
            acts.append(a)
        return (a, zs, acts) if cache else a

    @staticmethod
    def _bce(p, y):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, n_in = X.shape
        self._init_params(n_in)
        L = len(self.W)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(bb) for bb in self.b]; vb = [np.zeros_like(bb) for bb in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        rng = np.random.default_rng(self.seed + 1)
        # split interno train/val para EARLY STOPPING (evita sobreajuste)
        perm = rng.permutation(n)
        n_val = max(1, int(0.15 * n))
        vi, ti = perm[:n_val], perm[n_val:]
        Xtr, ytr, Xval, yval = X[ti], y[ti], X[vi], y[vi]
        self.loss_curve_, self.val_loss_curve_ = [], []
        best_val, best, bad, t = np.inf, None, 0, 0
        for _ep in range(self.epochs):
            order = rng.permutation(len(ti))
            for s in range(0, len(ti), self.batch):
                bi = order[s:s + self.batch]
                xb, yb = Xtr[bi], ytr[bi]
                p, zs, acts = self._forward(xb, cache=True)
                m = len(bi)
                dz = (p.reshape(-1) - yb).reshape(-1, 1) / m   # ∂BCE/∂z_salida = p−y
                t += 1
                for i in reversed(range(L)):
                    a_in = xb if i == 0 else acts[i - 1]
                    gW = a_in.T @ dz + self.l2 * self.W[i]
                    gb = dz.sum(0)
                    if i > 0:
                        dz = (dz @ self.W[i].T) * self._gelu_grad(zs[i - 1])
                    mW[i] = b1 * mW[i] + (1 - b1) * gW
                    vW[i] = b2 * vW[i] + (1 - b2) * (gW * gW)
                    self.W[i] -= self.lr * (mW[i] / (1 - b1 ** t)) / (np.sqrt(vW[i] / (1 - b2 ** t)) + eps)
                    mb[i] = b1 * mb[i] + (1 - b1) * gb
                    vb[i] = b2 * vb[i] + (1 - b2) * (gb * gb)
                    self.b[i] -= self.lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
            tr_loss = self._bce(self._forward(Xtr).reshape(-1), ytr)
            val_loss = self._bce(self._forward(Xval).reshape(-1), yval)
            self.loss_curve_.append(tr_loss)
            self.val_loss_curve_.append(val_loss)
            if val_loss < best_val - 1e-4:                 # mejora → guardar mejores pesos
                best_val = val_loss
                best = ([w.copy() for w in self.W], [bb.copy() for bb in self.b])
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:                   # sin mejorar → parar
                    break
        if best is not None:                               # restaurar mejores pesos
            self.W, self.b = best
        # compatibilidad con la extracción de artefactos (estilo sklearn)
        self.coefs_ = self.W
        self.intercepts_ = self.b
        self.n_iter_ = len(self.loss_curve_)
        self.best_val_loss_ = round(float(best_val), 4)
        return self

    def predict_proba(self, X):
        p = self._forward(np.asarray(X, dtype=float)).reshape(-1)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ── PRUEBA de neuronas muertas (verificación pedida) ──
    def hidden_activations(self, X):
        a = np.asarray(X, dtype=float)
        outs = []
        for i in range(len(self.W) - 1):          # solo capas ocultas
            a = self._gelu(a @ self.W[i] + self.b[i])
            outs.append(a)
        return outs

    def dead_neuron_report(self, X, tol=1e-3):
        """Una neurona está 'muerta' si su activación es ~constante (varianza≈0)
        para todas las entradas → no aporta. Con GELU esto debería ser 0."""
        layers = []
        for li, A in enumerate(self.hidden_activations(X)):
            std = A.std(axis=0)
            dead = int(np.sum(std < tol))
            layers.append({
                "layer": li + 1, "units": int(A.shape[1]), "dead": dead,
                "pct_dead": round(100.0 * dead / A.shape[1], 1),
                "mean_activation": round(float(np.abs(A).mean()), 4),
                "min_std": round(float(std.min()), 5),
            })
        total = sum(l["units"] for l in layers)
        deadt = sum(l["dead"] for l in layers)
        pct = round(100.0 * deadt / total, 1) if total else 0.0
        return {
            "layers": layers, "total_units": total, "total_dead": deadt,
            "pct_dead": pct,
            "healthy": pct <= 5.0,            # ≤5% inactivas = red sana
            "tol": tol,
            "note": ("GELU es suave y no anula neuronas a 0 como ReLU; verificamos "
                     "que ninguna quede inactiva (varianza ≈ 0 en todas las muestras)."),
        }


def _build_models():
    # Orden pedagógico: 1) Regresión Logística (base lineal interpretable)
    # 2) XGBoost (boosting por gradiente, usa gradiente y hessiano)
    # 3) Random Forest (bagging, controla el sobreajuste)
    # 4) Red Neuronal GELU+Sigmoide (aprende lo no-lineal sobre lo anterior).
    models = {"logit": LogisticRegression(max_iter=1000, solver="lbfgs")}
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    models["neural_net"] = GeluMLP(hidden=(64, 32), epochs=240, lr=0.01, seed=42)
    return models


PIPELINE_STEPS = [
    {"id": "ingest",   "label": "Ingesta de datos",        "icon": "i-upload"},
    {"id": "clean",    "label": "Limpieza de datos",        "icon": "i-refresh"},
    {"id": "engineer", "label": "Feature engineering",      "icon": "i-network"},
    {"id": "split",    "label": "Train / Test split",       "icon": "i-external"},
    {"id": "scale",    "label": "Escalado (StandardScaler)","icon": "i-shield"},
    {"id": "train_lr", "label": "Regresion Logistica",      "icon": "i-cpu"},
    {"id": "train_xgb","label": "XGBoost (gradiente+hessiano)", "icon": "i-cpu"},
    {"id": "train_rf", "label": "Random Forest",            "icon": "i-cpu"},
    {"id": "train_nn", "label": "Red Neuronal GELU+Sigmoide",   "icon": "i-cpu"},
    {"id": "evaluate", "label": "Evaluacion y metricas",    "icon": "i-shield"},
    {"id": "persist",  "label": "Persistir modelo (.pkl)",  "icon": "i-lock"},
]


def _extract_learning(models: dict, columns: list) -> dict:
    """Extrae artefactos interpretables tras entrenar:
      - nn_loss_curve: pérdida por iteración de la red (¿realmente aprende?).
      - nn_arch: capas + nº de parámetros (pesos + sesgos).
      - logit_coef: coeficientes β del logit = PESO y SIGNO esperado por variable.
      - logit_intercept: el sesgo β₀.
      - importance: importancia por variable (Random Forest, agregando one-hot).
    """
    out = {}
    nn = models.get("neural_net")
    if nn is not None and hasattr(nn, "loss_curve_"):
        lc = [round(float(v), 4) for v in nn.loss_curve_]
        out["nn_loss_curve"] = lc
        out["nn_final_loss"] = (lc[-1] if lc else None)
        out["nn_iters"] = len(lc)
        if hasattr(nn, "val_loss_curve_"):
            out["nn_val_loss_curve"] = [round(float(v), 4) for v in nn.val_loss_curve_]
        out["nn_activation"] = "GELU (ocultas) + Sigmoide (salida)"
    if nn is not None and hasattr(nn, "coefs_"):
        layers = [int(nn.coefs_[0].shape[0])] + [int(c.shape[1]) for c in nn.coefs_]
        n_params = int(sum(c.size for c in nn.coefs_) + sum(b.size for b in nn.intercepts_))
        out["nn_arch"] = {"layers": layers, "n_params": n_params}

    logit = models.get("logit")
    if logit is not None and hasattr(logit, "coef_"):
        pairs = [{"feature": c, "label": _friendly(c), "coef": round(float(w), 3)}
                 for c, w in zip(columns, logit.coef_[0])]
        pairs.sort(key=lambda d: abs(d["coef"]), reverse=True)
        out["logit_coef"] = pairs[:14]
        out["logit_intercept"] = round(float(logit.intercept_[0]), 3)

    rf = models.get("random_forest")
    if rf is not None and hasattr(rf, "feature_importances_"):
        agg = {}
        for col, imp in zip(columns, rf.feature_importances_):
            base = col
            for b in (RAW_NUM + RAW_CAT):
                if col == b or col.startswith(b + "_"):
                    base = b
                    break
            agg[base] = agg.get(base, 0.0) + float(imp)
        imp_list = [{"feature": k, "label": FRIENDLY.get(k, k), "importance": round(v, 4)}
                    for k, v in agg.items()]
        imp_list.sort(key=lambda d: d["importance"], reverse=True)
        out["importance"] = imp_list
    return out


def train_and_persist(seed: int = 42, df: "pd.DataFrame" = None,
                      source: str = "seed",
                      on_progress=None) -> dict:
    """Entrena los 4 modelos y persiste el bundle a disco.

    - Sin argumentos -> usa el dataset semilla sintetico (comportamiento original).
    - df != None     -> el admin subio un dataset real (mismo esquema de columnas).
    - on_progress(step_id, step_idx, total, detail) -> callback para barra de progreso.
    Train/test split 80/20 para metricas honestas (accuracy + AUC sobre test).
    """
    total = len(PIPELINE_STEPS)
    def _p(step_id, detail=""):
        if on_progress:
            idx = next((i for i, s in enumerate(PIPELINE_STEPS) if s["id"] == step_id), 0)
            on_progress(step_id, idx, total, detail)

    _p("ingest", f"Cargando dataset ({source})")
    if df is None:
        df = generate_seed(seed=seed)
    df = df.copy()

    _p("clean", f"{len(df)} filas, {len(df.columns)} columnas")
    y = df[TARGET].astype(int).values
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    _p("engineer", f"One-hot encoding de {len(RAW_CAT)} variables categoricas")
    X = _encode(df)
    columns = list(X.columns)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score

    _p("split", f"80% train / 20% test (stratified)")
    strat = y if len(np.unique(y)) == 2 and np.bincount(y).min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=strat)

    _p("scale", f"StandardScaler sobre {X.shape[1]} features")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    scaler_full = StandardScaler().fit(X.values)

    models = _build_models()
    model_keys = list(models.keys())
    step_map = {"logit": "train_lr", "random_forest": "train_rf",
                "xgboost": "train_xgb", "neural_net": "train_nn"}
    friendly = {"logit": "Regresion Logistica", "random_forest": "Random Forest",
                "xgboost": "XGBoost", "neural_net": "Red Neuronal"}
    metrics = {}
    for name, m in models.items():
        _p(step_map.get(name, "train_lr"), f"Entrenando {friendly.get(name, name)}...")
        m.fit(Xtr_s, ytr)
        try:
            proba = m.predict_proba(Xte_s)[:, 1]
            auc = float(roc_auc_score(yte, proba))
        except Exception:
            auc = 0.5
        try:
            acc = float(accuracy_score(yte, m.predict(Xte_s)))
        except Exception:
            acc = 0.0
        m.fit(scaler_full.transform(X.values), y)
        metrics[name] = {"auc": round(auc, 3), "accuracy": round(acc, 3)}

    scaler = scaler_full

    # ── Artefactos de aprendizaje (verificar/explicar: curva de pérdida,
    #    pesos y signos del logit, importancia de variables, arquitectura NN) ──
    learning = _extract_learning(models, columns)
    # Prueba de neuronas muertas sobre TODO el dataset escalado (verificación pedida)
    nn = models.get("neural_net")
    if nn is not None and hasattr(nn, "dead_neuron_report"):
        try:
            learning["dead_neurons"] = nn.dead_neuron_report(scaler.transform(X.values))
        except Exception:
            pass

    _p("evaluate", "Calculando metricas finales y rangos")
    ranges = {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                  "mean": float(X[c].mean()), "median": float(X[c].median())}
              for c in columns}

    import datetime as _dt
    _p("persist", "Guardando jnus_bundle.pkl")
    bundle = {
        "models": models, "scaler": scaler, "columns": columns,
        "ranges": ranges, "metrics": metrics,
        "raw_num": RAW_NUM, "raw_cat": RAW_CAT,
        "approval_rate": float(y.mean()),
        "class_distribution": {"aprobado": n_pos, "rechazado": n_neg},
        "n_features": int(X.shape[1]),
        "version": _dt.datetime.now().strftime("v%Y.%m.%d-%H%M"),
        "dataset_size": int(len(df)),
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "learning": learning,
    }
    if HAS_JOBLIB:
        joblib.dump(bundle, BUNDLE_PATH)
    return bundle


_TARGET_MAP = {
    "1": 1, "0": 0, "si": 1, "sí": 1, "no": 0, "yes": 1, "true": 1, "false": 0,
    "aprobado": 1, "aprobada": 1, "rechazado": 0, "rechazada": 0, "negado": 0,
    "approved": 1, "denied": 0, "default": 0, "pago": 1, "impago": 0, "y": 1, "n": 0,
}


def _normalize_target(s: "pd.Series") -> "pd.Series":
    """Convierte la columna objetivo de datos REALES a 0/1 admitiendo varias
    representaciones (Sí/No, Aprobado/Rechazado, true/false, 1/0, 0.0..1.0…)."""
    def conv(v):
        if pd.isna(v):
            return np.nan
        sv = str(v).strip().lower()
        if sv in _TARGET_MAP:
            return _TARGET_MAP[sv]
        try:
            return 1 if float(sv) >= 0.5 else 0
        except Exception:
            return np.nan
    return s.map(conv)


def retrain_from_dataframe(df: "pd.DataFrame", source: str = "upload") -> dict:
    """Reentrena el motor de PRODUCCIÓN desde un dataset del admin (datos REALES)
    y recarga el singleton para que /app use el nuevo modelo inmediatamente.

    Limpia los datos reales de forma robusta: normaliza el target a 0/1, fuerza
    numéricos en las columnas numéricas (NaN→mediana) y rellena categóricas vacías.
    """
    # Validar columnas mínimas requeridas
    required = set(RAW_NUM + RAW_CAT + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(sorted(missing)))

    df = df.copy()
    # Target → 0/1 robusto
    df[TARGET] = _normalize_target(df[TARGET])
    df = df.dropna(subset=[TARGET])
    if df.empty:
        raise ValueError("No se pudo interpretar la columna 'aprobado' (usa 1/0, Sí/No o Aprobado/Rechazado).")
    df[TARGET] = df[TARGET].astype(int)
    if df[TARGET].nunique() < 2:
        raise ValueError("El dataset debe incluir casos aprobados Y rechazados (ambas clases).")
    # Numéricos robustos
    for c in RAW_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[RAW_NUM] = df[RAW_NUM].fillna(df[RAW_NUM].median(numeric_only=True))
    # Categóricas: a texto, vacíos → "NA"
    for c in RAW_CAT:
        df[c] = df[c].astype(str).replace({"nan": "NA", "None": "NA", "": "NA"}).fillna("NA")

    bundle = train_and_persist(df=df, source=source)
    ENGINE.bundle = bundle  # hot-reload del modelo en memoria
    return bundle


# columnas que un CSV/Excel/SAV del admin debe contener
REQUIRED_COLUMNS = RAW_NUM + RAW_CAT + [TARGET]


# ──────────────────────────────────────────────────────────────────────────────
# 3) MOTOR DE INFERENCIA (singleton)
# ──────────────────────────────────────────────────────────────────────────────
class JanusEngine:
    def __init__(self):
        self.bundle = None

    def ready(self) -> bool:
        return self.bundle is not None

    def bootstrap(self):
        """Carga modelos persistidos o los entrena si no existen."""
        if HAS_JOBLIB and os.path.exists(BUNDLE_PATH):
            try:
                self.bundle = joblib.load(BUNDLE_PATH)
                return self.bundle
            except Exception:
                pass
        self.bundle = train_and_persist()
        return self.bundle

    # ── construir vector de features desde el formulario amigable ──
    def _row(self, payload: dict) -> pd.DataFrame:
        d = {
            "edad": float(payload.get("edad", 35)),
            "ingresos_mensuales": float(payload.get("ingresos_mensuales", 800)),
            "cargas_familiares": float(payload.get("cargas_familiares", 0)),
            "creditos_activos": float(payload.get("creditos_activos", 0)),
            "sexo": payload.get("sexo", "Masculino"),
            "educacion": payload.get("educacion", "Secundaria"),
            "historial_pagos": payload.get("historial_pagos", "Bueno"),
            "institucion": payload.get("institucion", "Banco Pichincha"),
            "tipo_credito": payload.get("tipo_credito", "Personal"),
            "situacion_laboral": payload.get("situacion_laboral", "Empleado Privado"),
        }
        return pd.DataFrame([d])

    def _prob_all(self, Xs) -> dict:
        out = {}
        for name, m in self.bundle["models"].items():
            try:
                out[name] = float(m.predict_proba(Xs)[0, 1])
            except Exception:
                out[name] = 0.5
        return out

    def score(self, payload: dict) -> dict:
        if not self.ready():
            self.bootstrap()
        b = self.bundle
        df = self._row(payload)
        X = _encode(df, columns=b["columns"])
        Xs = b["scaler"].transform(X.values)

        per_model = self._prob_all(Xs)
        prob = float(np.mean(list(per_model.values())))  # ensemble

        # ── XAI: contribuciones del modelo logístico (interpretable) ──
        logit = b["models"]["logit"]
        coefs = logit.coef_[0]
        contribs = []
        for col, coef, val in zip(b["columns"], coefs, Xs[0]):
            c = float(coef * val)
            if abs(c) < 1e-6:
                continue
            contribs.append({"feature": col, "label": _friendly(col),
                             "impact": c})
        contribs.sort(key=lambda d: abs(d["impact"]), reverse=True)
        positives = [c for c in contribs if c["impact"] > 0][:5]
        negatives = [c for c in contribs if c["impact"] < 0][:5]

        # ── Recomendaciones what-if (sobre el ensemble) ──
        recs = self._recommendations(payload, prob)

        # ── Clasificación de riesgo ──
        if prob >= 0.66:
            risk, risk_label, risk_color = "alta", "Excelente", "#22C55E"
        elif prob >= 0.40:
            risk, risk_label, risk_color = "media", "Moderada", "#D4AF37"
        else:
            risk, risk_label, risk_color = "baja", "Riesgosa", "#EF4444"

        verdict = ("¡Felicidades! Tu perfil tiene una alta probabilidad de aprobación."
                   if prob >= 0.66 else
                   "Tu perfil es viable, pero puedes mejorarlo para asegurar la aprobación."
                   if prob >= 0.40 else
                   "Tu perfil presenta riesgo. Sigue las recomendaciones para mejorar.")

        return {
            "ok": True,
            "probability": round(prob, 4),
            "percent": round(prob * 100, 1),
            "risk": risk, "risk_label": risk_label, "risk_color": risk_color,
            "verdict": verdict,
            "per_model": {k: round(v * 100, 1) for k, v in per_model.items()},
            "positive_factors": [
                {"label": c["label"], "weight": round(min(1.0, abs(c["impact"]) / 3), 2)}
                for c in positives
            ],
            "negative_factors": [
                {"label": c["label"], "weight": round(min(1.0, abs(c["impact"]) / 3), 2)}
                for c in negatives
            ],
            "recommendations": recs,
            "model_metrics": b["metrics"],
        }

    def _recommendations(self, payload: dict, base_prob: float) -> list:
        """Análisis what-if real: prueba mejoras y mide el cambio de probabilidad."""
        recs = []
        b = self.bundle

        def prob_of(mod_payload):
            X = _encode(self._row(mod_payload), columns=b["columns"])
            Xs = b["scaler"].transform(X.values)
            return float(np.mean(list(self._prob_all(Xs).values())))

        # 1) reducir créditos activos
        ca = int(float(payload.get("creditos_activos", 0)))
        if ca >= 1:
            alt = dict(payload); alt["creditos_activos"] = max(0, ca - 1)
            d = prob_of(alt) - base_prob
            if d > 0.01:
                recs.append({
                    "icon": "💳",
                    "text": f"Si reduces tus créditos activos de {ca} a {ca-1}, tu aprobación sube ~{d*100:.0f}%.",
                    "gain": round(d * 100, 1)})

        # 2) mejorar historial de pagos
        hp = payload.get("historial_pagos", "Bueno")
        if hp in PAYMENT_HISTORY and PAYMENT_HISTORY.index(hp) < 3:
            nxt = PAYMENT_HISTORY[PAYMENT_HISTORY.index(hp) + 1]
            alt = dict(payload); alt["historial_pagos"] = nxt
            d = prob_of(alt) - base_prob
            if d > 0.01:
                recs.append({
                    "icon": "📅",
                    "text": f"Mantener tus pagos al día (historial «{nxt}») aumentaría tu aprobación ~{d*100:.0f}%.",
                    "gain": round(d * 100, 1)})

        # 3) incrementar ingresos (informativo)
        ing = float(payload.get("ingresos_mensuales", 800))
        alt = dict(payload); alt["ingresos_mensuales"] = ing * 1.25
        d = prob_of(alt) - base_prob
        if d > 0.01:
            recs.append({
                "icon": "💵",
                "text": f"Demostrar ingresos un 25% mayores elevaría tu aprobación ~{d*100:.0f}%.",
                "gain": round(d * 100, 1)})

        recs.sort(key=lambda r: r["gain"], reverse=True)
        if not recs:
            recs.append({"icon": "✅",
                         "text": "Tu perfil ya está bien optimizado. ¡Mantén tus buenos hábitos financieros!",
                         "gain": 0})
        return recs[:4]

    def options(self) -> dict:
        return {
            "credit_types": CREDIT_TYPES,
            "employment": EMPLOYMENT,
            "education": EDUCATION,
            "payment_history": PAYMENT_HISTORY,
            "institutions": INSTITUTIONS,
            "sex": SEX,
        }


# singleton global
ENGINE = JanusEngine()
