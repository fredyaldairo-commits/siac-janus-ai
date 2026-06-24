#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JNUS AI · Limpiador de datasets
================================
Limpia y normaliza un dataset (CSV / Excel / SPSS) ANTES de subirlo al panel
/admin para reentrenar los modelos. NO modifica el backend ni los modelos:
es una herramienta independiente que prepara los datos.

Uso:
    python tools/limpiar_dataset.py  ruta/al/dataset.csv
    python tools/limpiar_dataset.py  "C:/Users/USER/Downloads/datos.xlsx"

Salida:
    Crea  <nombre>_limpio.csv  listo para subir en /admin.
"""
from __future__ import annotations
import sys
import os
import re

import pandas as pd

# Mismo esquema que el motor (engine.py) — fuente de verdad
RAW_NUM = ["edad", "ingresos_mensuales", "cargas_familiares", "creditos_activos"]
RAW_CAT = ["sexo", "educacion", "historial_pagos", "institucion",
           "tipo_credito", "situacion_laboral"]
TARGET = "aprobado"
REQUIRED = RAW_NUM + RAW_CAT + [TARGET]

# Variantes comunes de nombres de columna → nombre canónico
ALIASES = {
    "edad": ["edad", "age", "años", "anios"],
    "ingresos_mensuales": ["ingresos_mensuales", "ingresos", "ingreso", "salario", "income", "sueldo"],
    "cargas_familiares": ["cargas_familiares", "cargas", "dependientes", "hijos", "carga_familiar"],
    "creditos_activos": ["creditos_activos", "creditos", "prestamos_activos", "deudas_activas", "num_creditos"],
    "sexo": ["sexo", "genero", "género", "sex", "gender"],
    "educacion": ["educacion", "educación", "nivel_educativo", "estudios", "education"],
    "historial_pagos": ["historial_pagos", "historial", "historial_pago", "payment_history", "buro"],
    "institucion": ["institucion", "institución", "banco", "entidad", "cooperativa"],
    "tipo_credito": ["tipo_credito", "tipo_crédito", "tipo", "producto", "credito"],
    "situacion_laboral": ["situacion_laboral", "situación_laboral", "empleo", "trabajo", "ocupacion", "laboral"],
    "aprobado": ["aprobado", "aprobada", "resultado", "target", "approved", "decision", "estado"],
}

# Valores que significan "aprobado=1" / "aprobado=0"
TRUE_SET = {"1", "si", "sí", "yes", "true", "aprobado", "aprobada", "approved", "y", "v", "verdadero"}
FALSE_SET = {"0", "no", "false", "rechazado", "rechazada", "denied", "n", "f", "falso"}


def _norm(s: str) -> str:
    s = str(s).replace("﻿", "").replace("​", "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    return s


def load(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        # Detecta separador automáticamente; utf-8-sig elimina el BOM inicial
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, encoding="utf-8-sig")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".sav":
        try:
            import pyreadstat
            df, _ = pyreadstat.read_sav(path)
            return df
        except ImportError:
            sys.exit("[X] Para .SAV necesitas: pip install pyreadstat")
    sys.exit(f"[X] Formato no soportado: {ext}  (usa .csv, .xlsx, .xls o .sav)")


def remap_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Renombra columnas a los nombres canónicos usando los alias."""
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    rename = {}
    for canon, variants in ALIASES.items():
        vset = {_norm(v) for v in variants}
        for col in df.columns:
            if col in vset and col != canon:
                rename[col] = canon
                break
    df = df.rename(columns=rename)
    return df, rename


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    notes = []
    n0 = len(df)

    # 1) Quitar filas/columnas totalmente vacías
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if len(df) < n0:
        notes.append(f"Filas vacías eliminadas: {n0 - len(df)}")

    # 2) Eliminar duplicados
    dups = df.duplicated().sum()
    if dups:
        df = df.drop_duplicates()
        notes.append(f"Filas duplicadas eliminadas: {dups}")

    # 3) Limpiar espacios en texto
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": pd.NA, "": pd.NA, "NaN": pd.NA, "None": pd.NA})

    # 4) Coaccionar numéricos
    for c in RAW_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # rellenar nulos con la mediana
            if df[c].isna().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)
                notes.append(f"'{c}': nulos rellenados con mediana ({med:g})")

    # 5) Normalizar el target a 0/1
    if TARGET in df.columns:
        def to_bin(v):
            s = _norm(v)
            if s in TRUE_SET:
                return 1
            if s in FALSE_SET:
                return 0
            try:
                return 1 if float(s) >= 0.5 else 0
            except Exception:
                return pd.NA
        df[TARGET] = df[TARGET].map(to_bin)
        bad = df[TARGET].isna().sum()
        if bad:
            df = df.dropna(subset=[TARGET])
            notes.append(f"'{TARGET}': {bad} filas con valor no interpretable eliminadas")
        df[TARGET] = df[TARGET].astype(int)

    # 6) Rellenar categóricas vacías con la moda
    for c in RAW_CAT:
        if c in df.columns and df[c].isna().any():
            mode = df[c].mode()
            fill = mode.iloc[0] if len(mode) else "Desconocido"
            df[c] = df[c].fillna(fill)
            notes.append(f"'{c}': nulos rellenados con '{fill}'")

    return df, notes


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        path = input("Arrastra aquí tu archivo y pulsa Enter:\n> ").strip().strip('"')
    else:
        path = sys.argv[1].strip().strip('"')

    if not os.path.exists(path):
        sys.exit(f"[X] No existe el archivo: {path}")

    print("=" * 60)
    print(" JNUS AI · Limpiador de datasets")
    print("=" * 60)
    print(f" Archivo: {path}")

    df = load(path)
    print(f" Filas originales: {len(df)}  ·  Columnas: {len(df.columns)}")

    df, rename = remap_columns(df)
    if rename:
        print("\n [+] Columnas renombradas a formato JNUS:")
        for k, v in rename.items():
            print(f"     {k}  ->  {v}")

    df, notes = clean(df)

    # Reporte de columnas requeridas
    present = [c for c in REQUIRED if c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    print(f"\n [i] Columnas requeridas presentes: {len(present)}/{len(REQUIRED)}")
    if missing:
        print(" [!] FALTAN columnas (añádelas antes de subir a /admin):")
        for c in missing:
            print(f"       - {c}")

    if notes:
        print("\n [+] Limpieza aplicada:")
        for n in notes:
            print(f"     - {n}")

    # Guardar
    base = os.path.splitext(path)[0]
    out = base + "_limpio.csv"
    # Reordenar columnas: requeridas primero
    cols = present + [c for c in df.columns if c not in present]
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f" [OK] Dataset limpio guardado en:\n      {out}")
    print(f"      Filas finales: {len(df)}  ·  Columnas: {len(df.columns)}")
    if not missing:
        print("      LISTO para subir en  /admin  ->  Entrenar y publicar")
    else:
        print("      Completa las columnas faltantes y vuelve a ejecutar.")
    print("=" * 60)


if __name__ == "__main__":
    main()
