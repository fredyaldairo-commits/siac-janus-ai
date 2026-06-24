#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JNUS AI · Generador de reportes del modelo
==========================================
Genera un INFORME imprimible (HTML -> PDF) del modelo en producción:
  - Métricas de rendimiento de los 4 modelos
  - Composición del dataset
  - Análisis de sesgo / equidad (opcional, si se aporta el dataset)
  - Documentación técnica del pipeline

NO modifica el backend ni los modelos: solo LEE el bundle ya entrenado.

Uso:
    python tools/reporte_modelo.py
    python tools/reporte_modelo.py  dataset_de_entrenamiento.csv   # añade análisis de sesgo

Salida:
    reporte_modelo.html   (ábrelo y usa "Imprimir -> Guardar como PDF")
"""
from __future__ import annotations
import os
import sys
import datetime

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "models", "janus_bundle.pkl")
NAMES = {"logit": "Regresión Logística", "random_forest": "Random Forest",
         "xgboost": "XGBoost", "neural_net": "Red Neuronal"}


def load_bundle():
    if not os.path.exists(BUNDLE):
        sys.exit(f"[X] No existe el modelo entrenado: {BUNDLE}\n    Entrena primero desde /admin.")
    return joblib.load(BUNDLE)


def bias_section(csv_path: str) -> str:
    """Calcula tasa de aprobación por grupo (sexo, institución) = análisis de equidad."""
    try:
        import pandas as pd
    except ImportError:
        return ""
    if not os.path.exists(csv_path):
        return f'<p style="color:#DC2626">No se encontró el dataset: {csv_path}</p>'
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "aprobado" not in df.columns:
        return '<p style="color:#DC2626">El dataset no tiene columna "aprobado".</p>'

    def to_bin(v):
        s = str(v).strip().lower()
        if s in {"1", "si", "sí", "yes", "true", "aprobado", "aprobada"}:
            return 1
        if s in {"0", "no", "false", "rechazado", "rechazada"}:
            return 0
        try:
            return 1 if float(s) >= 0.5 else 0
        except Exception:
            return None
    df["_y"] = df["aprobado"].map(to_bin)
    df = df.dropna(subset=["_y"])

    html = ""
    for col, titulo in [("sexo", "Equidad por sexo"), ("institucion", "Equidad por institución")]:
        if col not in df.columns:
            continue
        g = df.groupby(col)["_y"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        rows = ""
        rates = g["mean"].tolist()
        for idx, r in g.iterrows():
            pct = r["mean"] * 100
            bar = int(pct)
            rows += (f'<tr><td>{idx}</td>'
                     f'<td style="width:50%"><div style="background:#E5E9F0;border-radius:4px;height:12px">'
                     f'<div style="width:{bar}%;height:100%;border-radius:4px;background:#2563EB"></div></div></td>'
                     f'<td style="text-align:right;font-weight:700">{pct:.1f}%</td>'
                     f'<td style="text-align:right;color:#64748B">{int(r["count"])}</td></tr>')
        # Disparidad: diferencia entre el grupo más y menos favorecido
        disp = (max(rates) - min(rates)) * 100 if rates else 0
        flag = ("#16A34A", "Baja") if disp < 10 else ("#D97706", "Moderada") if disp < 20 else ("#DC2626", "Alta")
        html += (f'<h3>{titulo}</h3>'
                 f'<p style="color:#64748B;margin-bottom:6px">Disparidad entre grupos: '
                 f'<b style="color:{flag[0]}">{disp:.1f} pts ({flag[1]})</b> '
                 f'— diferencia en la tasa de aprobación entre el grupo más y menos favorecido.</p>'
                 f'<table class="t"><tr><th>Grupo</th><th>Tasa de aprobación</th><th></th><th>n</th></tr>{rows}</table>')
    return html or '<p style="color:#94A3B8">No se encontraron columnas de grupo para el análisis.</p>'


def main():
    b = load_bundle()
    metrics = b.get("metrics", {})
    best = max(metrics, key=lambda k: metrics[k].get("auc", 0)) if metrics else None
    cd = b.get("class_distribution", {}) or {}
    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    # Tabla de modelos
    mrows = ""
    for k, v in sorted(metrics.items(), key=lambda kv: kv[1].get("auc", 0), reverse=True):
        star = ' ★' if k == best else ''
        hi = ' style="background:#F7F0DC"' if k == best else ''
        mrows += (f'<tr{hi}><td>{NAMES.get(k, k)}{star}</td>'
                  f'<td style="text-align:right;font-weight:700">{v.get("auc", 0):.3f}</td>'
                  f'<td style="text-align:right">{v.get("accuracy", 0)*100:.1f}%</td></tr>')

    bias = bias_section(sys.argv[1].strip().strip('"')) if len(sys.argv) > 1 else (
        '<p style="color:#94A3B8">Para incluir el análisis de sesgo, ejecuta:<br/>'
        '<code>python tools/reporte_modelo.py tu_dataset.csv</code></p>')

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"/>
<title>Reporte del modelo · JNUS AI · {b.get('version','')}</title>
<style>
  @page{{margin:16mm}}
  *{{box-sizing:border-box;margin:0;font-family:'Segoe UI',system-ui,sans-serif}}
  body{{color:#0F172A;font-size:13px;line-height:1.55}}
  .hd{{display:flex;align-items:center;gap:14px;border-bottom:3px solid #C8A24B;padding-bottom:14px;margin-bottom:18px}}
  .hd .t{{font-size:22px;font-weight:800;letter-spacing:1px}}
  .hd .t span{{color:#C8A24B;font-size:13px}}
  .hd .meta{{margin-left:auto;text-align:right;color:#64748B;font-size:11px}}
  h2{{font-size:16px;margin:22px 0 6px;color:#0F172A}}
  h3{{font-size:13px;margin:14px 0 6px;color:#0F172A;border-left:3px solid #C8A24B;padding-left:8px}}
  .cards{{display:flex;gap:12px;margin:10px 0}}
  .c{{flex:1;background:#F4F6FA;border:1px solid #E5E9F0;border-radius:10px;padding:14px;text-align:center}}
  .c .v{{font-size:24px;font-weight:800}}.c .l{{font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:.5px}}
  table.t{{width:100%;border-collapse:collapse;margin-top:6px}}
  table.t th{{text-align:left;color:#64748B;font-size:10px;text-transform:uppercase;border-bottom:1px solid #E5E9F0;padding:6px}}
  table.t td{{padding:6px;border-bottom:1px solid #E5E9F0}}
  ol{{padding-left:20px}}ol li{{margin:3px 0;color:#334155}}
  .foot{{margin-top:24px;padding-top:12px;border-top:1px solid #E5E9F0;color:#94A3B8;font-size:10px;text-align:center}}
  code{{background:#F4F6FA;padding:2px 6px;border-radius:4px;font-size:11px}}
</style></head><body>
  <div class="hd">
    <div class="t">JNUS<span> AI</span>
      <div style="font-size:11px;color:#64748B;font-weight:600;letter-spacing:0">Reporte técnico del modelo de crédito</div></div>
    <div class="meta">Versión: {b.get('version','—')}<br/>Generado: {fecha}</div>
  </div>

  <h2>1. Resumen del modelo</h2>
  <div class="cards">
    <div class="c"><div class="v">{b.get('dataset_size',0):,}</div><div class="l">Registros</div></div>
    <div class="c"><div class="v">{b.get('n_features','—')}</div><div class="l">Features</div></div>
    <div class="c"><div class="v">{b.get('approval_rate',0)*100:.1f}%</div><div class="l">Tasa aprobación</div></div>
    <div class="c"><div class="v">{metrics.get(best,{}).get('auc',0):.3f}</div><div class="l">AUC (mejor)</div></div>
  </div>
  <p style="color:#64748B">Mejor algoritmo: <b style="color:#0F172A">{NAMES.get(best,best or '—')}</b> ·
     Aprobados: <b>{cd.get('aprobado','—')}</b> · Rechazados: <b>{cd.get('rechazado','—')}</b></p>

  <h2>2. Rendimiento por algoritmo</h2>
  <table class="t"><tr><th>Modelo</th><th style="text-align:right">AUC</th><th style="text-align:right">Accuracy</th></tr>{mrows}</table>

  <h2>3. Análisis de sesgo y equidad</h2>
  {bias}

  <h2>4. Documentación del pipeline</h2>
  <ol>
    <li><b>Ingesta:</b> dataset CSV/Excel/SPSS cargado desde /admin.</li>
    <li><b>Limpieza:</b> validación de columnas requeridas y tipos.</li>
    <li><b>Feature engineering:</b> one-hot encoding de variables categóricas.</li>
    <li><b>Split:</b> 80% entrenamiento / 20% prueba (estratificado).</li>
    <li><b>Escalado:</b> StandardScaler sobre las features.</li>
    <li><b>Entrenamiento:</b> 4 modelos (Logística, Random Forest, XGBoost, MLP).</li>
    <li><b>Evaluación:</b> AUC y Accuracy sobre el conjunto de prueba.</li>
    <li><b>Selección:</b> el modelo con mayor AUC se publica como principal.</li>
    <li><b>Publicación:</b> <code>janus_bundle.pkl</code> → /app usa el modelo en caliente.</li>
  </ol>
  <p style="color:#64748B;margin-top:8px"><b>Variables protegidas:</b> el atributo <code>sexo</code> debe vigilarse para
     evitar discriminación. Revisa la sección 3: una disparidad alta (&gt;20 pts) sugiere re-balancear el dataset.</p>

  <div class="foot">JNUS AI · Advanced Financial System · SIAC · Machala, Ecuador · Documento interno de administración</div>
  <script>window.onload=function(){{setTimeout(function(){{window.print();}},400);}}</script>
</body></html>"""

    out = os.path.join(ROOT, "reporte_modelo.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("=" * 60)
    print(" JNUS AI · Reporte del modelo generado")
    print("=" * 60)
    print(f" Archivo: {out}")
    print(" Abrelo en el navegador y usa: Imprimir -> Guardar como PDF")
    print("=" * 60)
    # Intentar abrir automáticamente
    try:
        import webbrowser
        webbrowser.open("file://" + out.replace("\\", "/"))
    except Exception:
        pass


if __name__ == "__main__":
    main()
