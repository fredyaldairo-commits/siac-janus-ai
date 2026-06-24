"""
JNUS AI · Capa de Base de Datos (SQLite)
=========================================
Persistencia de usuarios y evaluaciones para que el modelo aprenda con más
datos reales. Diseño:

  - SQLite puro (stdlib `sqlite3`): cero dependencias, funciona en Windows,
    Linux y Render. El archivo vive en `data/janus.db` (gitignored).
  - NO toca los modelos ni el pipeline de ML. Solo GUARDA lo que el usuario
    responde en la encuesta y la probabilidad que el modelo le devolvió.
  - Cada evaluación guarda EXACTAMENTE las columnas que `engine.py` necesita
    (RAW_NUM + RAW_CAT) para poder exportar un dataset y reentrenar.
  - El campo `aprobado` (target) se guarda como la predicción binaria del
    modelo (prob ≥ 0.5). El admin puede corregirlo con el resultado real
    (`aprobado_real`) para que el reentrenamiento use la verdad de campo.

Para producción persistente en Render conviene migrar a Postgres; la interfaz
de funciones de este módulo se mantiene igual.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.environ.get("JNUS_DB_PATH", os.path.join(DATA_DIR, "janus.db"))

# Columnas de la encuesta = esquema del motor (engine.RAW_NUM + RAW_CAT)
SURVEY_NUM = ["edad", "ingresos_mensuales", "cargas_familiares", "creditos_activos"]
SURVEY_CAT = ["sexo", "educacion", "historial_pagos", "institucion",
              "tipo_credito", "situacion_laboral"]
SURVEY_COLS = SURVEY_NUM + SURVEY_CAT

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Crea las tablas si no existen. Idempotente."""
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                visits      INTEGER DEFAULT 1,
                evaluations INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluaciones (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at         TEXT NOT NULL,
                user_id            TEXT,
                user_name          TEXT,
                edad               REAL,
                ingresos_mensuales REAL,
                cargas_familiares  REAL,
                creditos_activos   REAL,
                sexo               TEXT,
                educacion          TEXT,
                historial_pagos    TEXT,
                institucion        TEXT,
                tipo_credito       TEXT,
                situacion_laboral  TEXT,
                probability        REAL,
                percent            REAL,
                risk               TEXT,
                decision           TEXT,
                aprobado           INTEGER,
                aprobado_real      INTEGER,
                per_model          TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_eval_created ON evaluaciones(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_eval_user ON evaluaciones(user_id)")
        # Migración: columnas de cuenta real (email/contraseña/foto). Idempotente.
        _add_col(conn, "usuarios", "email", "TEXT")
        _add_col(conn, "usuarios", "password_hash", "TEXT")
        _add_col(conn, "usuarios", "avatar", "TEXT")
        _add_col(conn, "usuarios", "updated_at", "TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_user_email "
                     "ON usuarios(email) WHERE email IS NOT NULL")


def _add_col(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """Añade una columna solo si no existe (SQLite no soporta ADD COLUMN IF NOT EXISTS)."""
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ──────────────────────────────────────────────────────────────────────────────
# USUARIOS
# ──────────────────────────────────────────────────────────────────────────────
def register_user(name: str) -> Dict[str, Any]:
    """Registra (o reconoce) un usuario por nombre/apodo."""
    name = (name or "").strip()[:40]
    now = _now()
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM usuarios WHERE lower(name)=lower(?)", (name,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE usuarios SET last_seen=?, visits=visits+1 WHERE id=?",
                (now, row["id"]),
            )
            return {"id": row["id"], "name": name, "returning": True}
        uid = _dt.datetime.now().strftime("u%Y%m%d%H%M%S%f")
        conn.execute(
            "INSERT INTO usuarios (id,name,created_at,last_seen,visits,evaluations) "
            "VALUES (?,?,?,?,1,0)",
            (uid, name, now, now),
        )
        return {"id": uid, "name": name, "returning": False}


def list_users() -> List[Dict[str, Any]]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT id,name,email,created_at,last_seen,visits,evaluations "
            "FROM usuarios ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# CUENTAS REALES (email + contraseña + foto)
# ──────────────────────────────────────────────────────────────────────────────
def _public_user(row: sqlite3.Row) -> Dict[str, Any]:
    """Dict del usuario SIN el hash de contraseña (seguro para enviar al cliente)."""
    d = dict(row)
    d.pop("password_hash", None)
    d["has_password"] = bool(row["password_hash"]) if "password_hash" in row.keys() else False
    return d


def create_account(name: str, email: str, password: str) -> Dict[str, Any]:
    """Crea una cuenta real. Devuelve {ok, user} o {ok:False, error}."""
    name = (name or "").strip()[:40]
    email = (email or "").strip().lower()[:120]
    if not name:
        return {"ok": False, "error": "Escribe tu nombre."}
    if not _EMAIL_RE.match(email):
        return {"ok": False, "error": "Email no válido."}
    if len(password or "") < 6:
        return {"ok": False, "error": "La contraseña debe tener al menos 6 caracteres."}
    now = _now()
    with _lock, _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM usuarios WHERE email=?", (email,)
        ).fetchone()
        if exists:
            return {"ok": False, "error": "Ya existe una cuenta con ese email."}
        uid = _dt.datetime.now().strftime("u%Y%m%d%H%M%S%f")
        conn.execute(
            "INSERT INTO usuarios (id,name,email,password_hash,created_at,last_seen,"
            "visits,evaluations,updated_at) VALUES (?,?,?,?,?,?,1,0,?)",
            (uid, name, email, generate_password_hash(password), now, now, now),
        )
        row = conn.execute("SELECT * FROM usuarios WHERE id=?", (uid,)).fetchone()
    return {"ok": True, "user": _public_user(row)}


def authenticate(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verifica credenciales. Devuelve el usuario público o None."""
    email = (email or "").strip().lower()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM usuarios WHERE email=?", (email,)).fetchone()
        if not row or not row["password_hash"]:
            return None
        if not check_password_hash(row["password_hash"], password or ""):
            return None
        conn.execute(
            "UPDATE usuarios SET last_seen=?, visits=visits+1 WHERE id=?",
            (_now(), row["id"]),
        )
        row = conn.execute("SELECT * FROM usuarios WHERE id=?", (row["id"],)).fetchone()
    return _public_user(row)


def get_user(uid: str) -> Optional[Dict[str, Any]]:
    """Usuario público por id (para restaurar la sesión)."""
    if not uid:
        return None
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM usuarios WHERE id=?", (uid,)).fetchone()
    return _public_user(row) if row else None


def update_profile(uid: str, name: Optional[str] = None,
                   avatar: Optional[str] = None,
                   password: Optional[str] = None) -> Dict[str, Any]:
    """Actualiza nombre / foto (data-URI) / contraseña del usuario en sesión."""
    sets, vals = [], []
    if name is not None:
        name = name.strip()[:40]
        if not name:
            return {"ok": False, "error": "El nombre no puede estar vacío."}
        sets.append("name=?"); vals.append(name)
    if avatar is not None:
        if avatar and len(avatar) > 700_000:  # ~512KB de imagen en base64
            return {"ok": False, "error": "La foto es demasiado grande."}
        sets.append("avatar=?"); vals.append(avatar or None)
    if password is not None and password != "":
        if len(password) < 6:
            return {"ok": False, "error": "La contraseña debe tener al menos 6 caracteres."}
        sets.append("password_hash=?"); vals.append(generate_password_hash(password))
    if not sets:
        return {"ok": False, "error": "Nada que actualizar."}
    sets.append("updated_at=?"); vals.append(_now())
    vals.append(uid)
    with _lock, _connect() as conn:
        conn.execute(f"UPDATE usuarios SET {', '.join(sets)} WHERE id=?", vals)
        row = conn.execute("SELECT * FROM usuarios WHERE id=?", (uid,)).fetchone()
    if not row:
        return {"ok": False, "error": "Usuario no encontrado."}
    return {"ok": True, "user": _public_user(row)}


# ──────────────────────────────────────────────────────────────────────────────
# EVALUACIONES
# ──────────────────────────────────────────────────────────────────────────────
def save_evaluation(payload: Dict[str, Any], result: Dict[str, Any],
                    user: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Guarda una evaluación (encuesta + resultado del modelo).

    Nunca lanza: si algo falla, devuelve None para no romper el scoring.
    """
    try:
        prob = float(result.get("probability", 0.0))
        decision = ("APROBAR" if prob >= 0.66 else
                    "REVISAR" if prob >= 0.40 else "RECHAZAR")
        aprobado = 1 if prob >= 0.5 else 0
        per_model = json.dumps(result.get("per_model", {}), ensure_ascii=False)
        uid = (user or {}).get("id")
        uname = (user or {}).get("name")

        def _num(k, d=0.0):
            try:
                return float(payload.get(k, d))
            except Exception:
                return d

        def _txt(k, d=""):
            return str(payload.get(k, d))

        with _lock, _connect() as conn:
            cur = conn.execute("""
                INSERT INTO evaluaciones (
                    created_at,user_id,user_name,
                    edad,ingresos_mensuales,cargas_familiares,creditos_activos,
                    sexo,educacion,historial_pagos,institucion,tipo_credito,situacion_laboral,
                    probability,percent,risk,decision,aprobado,aprobado_real,per_model
                ) VALUES (?,?,?, ?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?,?)
            """, (
                _now(), uid, uname,
                _num("edad"), _num("ingresos_mensuales"),
                _num("cargas_familiares"), _num("creditos_activos"),
                _txt("sexo"), _txt("educacion"), _txt("historial_pagos"),
                _txt("institucion"), _txt("tipo_credito"), _txt("situacion_laboral"),
                prob, float(result.get("percent", prob * 100)),
                str(result.get("risk", "")), decision, aprobado, None, per_model,
            ))
            if uid:
                conn.execute(
                    "UPDATE usuarios SET evaluations=evaluations+1 WHERE id=?", (uid,))
            return cur.lastrowid
    except Exception:
        return None


def list_evaluations(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluaciones ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def stats() -> Dict[str, Any]:
    """Resumen para el panel admin."""
    with _lock, _connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM evaluaciones").fetchone()["c"]
        users = conn.execute("SELECT COUNT(*) c FROM usuarios").fetchone()["c"]
        appr = conn.execute(
            "SELECT COUNT(*) c FROM evaluaciones WHERE aprobado=1").fetchone()["c"]
        avg = conn.execute(
            "SELECT AVG(probability) a FROM evaluaciones").fetchone()["a"]
        labeled = conn.execute(
            "SELECT COUNT(*) c FROM evaluaciones WHERE aprobado_real IS NOT NULL"
        ).fetchone()["c"]
        # distribución por tipo de crédito (con aprobados para tasa por tipo)
        by_type = conn.execute(
            "SELECT tipo_credito t, COUNT(*) c, COALESCE(SUM(aprobado),0) a "
            "FROM evaluaciones GROUP BY tipo_credito ORDER BY c DESC"
        ).fetchall()
        # distribución por banda de riesgo (alta/media/baja)
        by_risk = conn.execute(
            "SELECT risk r, COUNT(*) c FROM evaluaciones "
            "WHERE risk IS NOT NULL AND risk != '' GROUP BY risk"
        ).fetchall()
        # tendencia: evaluaciones por día (últimos 14 días con datos)
        by_day = conn.execute(
            "SELECT substr(created_at,1,10) d, COUNT(*) c, "
            "COALESCE(AVG(probability),0) p FROM evaluaciones "
            "GROUP BY substr(created_at,1,10) ORDER BY d DESC LIMIT 14"
        ).fetchall()
    return {
        "total_evaluaciones": total,
        "total_usuarios": users,
        "aprobados": appr,
        "rechazados": total - appr,
        "approval_rate": round((appr / total) if total else 0.0, 3),
        "avg_probability": round(avg or 0.0, 3),
        "etiquetados_real": labeled,
        "por_tipo_credito": [
            {"tipo": r["t"], "count": r["c"], "aprobados": r["a"]} for r in by_type
        ],
        "por_riesgo": [{"riesgo": r["r"], "count": r["c"]} for r in by_risk],
        "tendencia": [
            {"dia": r["d"], "count": r["c"], "prob": round(r["p"], 3)}
            for r in reversed(by_day)
        ],
    }


def set_real_label(eval_id: int, aprobado_real: int) -> bool:
    """El admin marca el resultado real (0/1) de una evaluación → reentrenamiento
    supervisado con verdad de campo."""
    try:
        with _lock, _connect() as conn:
            conn.execute(
                "UPDATE evaluaciones SET aprobado_real=? WHERE id=?",
                (int(aprobado_real), int(eval_id)),
            )
        return True
    except Exception:
        return False


def export_dataframe(use_real_when_available: bool = True):
    """Devuelve un pandas.DataFrame con el esquema EXACTO de engine.py
    (SURVEY_COLS + 'aprobado') listo para reentrenar el modelo.

    - Si `use_real_when_available`, usa `aprobado_real` cuando el admin lo
      etiquetó; si no, usa la predicción binaria del modelo (`aprobado`).
    """
    import pandas as pd
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT edad,ingresos_mensuales,cargas_familiares,creditos_activos,"
            "sexo,educacion,historial_pagos,institucion,tipo_credito,situacion_laboral,"
            "aprobado,aprobado_real FROM evaluaciones"
        ).fetchall()
    data = []
    for r in rows:
        d = {c: r[c] for c in SURVEY_COLS}
        real = r["aprobado_real"]
        d["aprobado"] = int(real) if (use_real_when_available and real is not None) else int(r["aprobado"] or 0)
        data.append(d)
    return pd.DataFrame(data, columns=SURVEY_COLS + ["aprobado"])


# Inicializa al importar (seguro/idempotente).
try:
    init_db()
except Exception:
    pass
