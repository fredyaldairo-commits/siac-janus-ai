# JNUS · Advanced Financial System

Plataforma de análisis de riesgo crediticio con IA + econometría.
Frontend en HTML/CSS/JS (sin frameworks) + Backend Flask con scikit-learn, XGBoost y statsmodels.

## 🚀 Cómo ejecutar (Windows)

Doble clic a **`run.bat`**.

Lo que hace automáticamente:
1. Crea un entorno virtual `.venv\`
2. Instala las dependencias de `requirements.txt`
3. Lanza Flask en `http://127.0.0.1:5000`
4. Abre el navegador en esa URL

### Manual (cualquier OS)
```bash
python -m venv .venv
.venv\Scripts\activate     # (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Abre `http://127.0.0.1:5000`.

## 📁 Estructura

```
JNUS/
├── app.py                  # Backend Flask + pipeline ML
├── requirements.txt        # Dependencias Python
├── run.bat                 # Lanzador Windows
├── templates/
│   └── index.html          # Frontend completo (HTML+CSS+JS+Chart.js)
├── static/                 # (opcional: assets estáticos)
└── uploads/                # CSV/Excel subidos por el usuario
```

## 🔄 Pipeline end-to-end

| Paso | Endpoint | Acción |
|---|---|---|
| 01 | `POST /api/upload` | Recibe CSV/Excel, autodetecta separador y encoding |
| 02 | `POST /api/preprocess` | Limpieza, imputación, one-hot, train/test split, escalado |
| 03 | `POST /api/train` | Entrena el modelo y devuelve métricas + visualizaciones |
| —  | `GET /api/health` | Estado del servidor + librerías disponibles |
| —  | `POST /api/reset` | Limpia el estado de la sesión |

## 🤖 Modelos soportados

- **Logit** — `LogisticRegression`
- **Probit** — `statsmodels.Probit` (fallback a Logit si no hay statsmodels)
- **Random Forest** — `RandomForestClassifier(n=200)`
- **XGBoost** — `XGBClassifier` (fallback a RF si no hay xgboost)
- **Red Neuronal** — `MLPClassifier(64,32)`

## 📊 Métricas calculadas

Accuracy · AUC-ROC · Gini · Precisión · Recall · F1 · Matriz de confusión · Importancia de variables · Curva ROC · Histograma de probabilidades.

## ✅ Errores corregidos respecto al HTML original

1. **No existía backend** — Se creó `app.py` Flask completo con los 3 endpoints que el frontend invocaba.
2. **Upload CSV/Excel fallaba** — Detección automática de separador (`,;|\t`) y encoding (`utf-8/latin-1`), soporte real `.xlsx` vía openpyxl.
3. **Variables categóricas mal codificadas** — Normalización Sí/No/Yes/True → 0/1 + `get_dummies(drop_first=True)` para evitar dummy trap.
4. **Valores nulos rompían el entrenamiento** — Imputación con mediana (numéricas) y moda (categóricas).
5. **Target no-binario** — Validación que sea binario y mapeo automático de 2 clases a 0/1.
6. **Modelos no entrenaban** — Pipeline completo: scaler + fit + predict_proba con fallback decisión-función.
7. **Probit no funcionaba** — Implementado con statsmodels MLE; fallback a Logit si falta la librería.
8. **XGBoost crashea sin librería** — Fallback automático a Random Forest.
9. **Gráficos vacíos** — Backend envía `y_prob_sample`, `y_test_sample`, `roc_points` y `feature_importance` siempre.
10. **JSON serialization errors** — Función `safe_jsonable()` convierte numpy/pandas a tipos nativos.
11. **Estado se perdía entre requests** — `STATE` global mantiene df + scaler entre pasos del pipeline.
12. **Sin logs visuales** — Cada paso del backend agrega entradas con timestamp al log y el frontend las pinta como timeline.
13. **Sin manejo de errores en UI** — Toasts con tipo `error/ok` y validación de extensión en cliente antes del POST.
14. **Stratify rompía con clases <2** — Check `value_counts().min() >= 2` antes de pasar a `train_test_split`.
15. **Columnas constantes rompían el modelo** — Se eliminan automáticamente tras one-hot.

## 🎨 Rediseño visual

Tema **midnight + gold** inspirado en banca premium (referencia: logo JNUS columna dorada + circuito azul):

- Sidebar fija con steps numerados (01-06)
- Topbar con título serif (Cormorant Garamond) + crumb
- Cards con borde dorado superior y sombra suave
- Métricas con franja vertical dorada
- Botones dorados (`btn-gold`) + ghost outline
- Drag & drop con halo radial dorado
- Tipografía: Inter (body) · Cormorant Garamond (display) · JetBrains Mono (código)
- Paleta: `#070b14` fondo, `#c89b3c` oro, `#5a8cff` azul, `#3dd68a` verde
