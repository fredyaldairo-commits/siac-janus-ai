# 🚀 Desplegar SIAC · JANUS AI en la web (24/7)

El proyecto está listo para producción. Ya no depende de tu terminal ni de tu PC.

## Opción A — Render.com (Recomendado, gratis)

1. **Crea cuenta** en https://render.com (entra con GitHub).
2. Sube el proyecto a un repo de GitHub:
   ```bash
   cd C:\Users\USER\Downloads\JNUS
   git init
   git add .
   git commit -m "JANUS AI initial"
   gh repo create siac-janus-ai --public --source=. --push
   ```
3. En Render → **New +** → **Blueprint** → selecciona tu repo. Render detectará `render.yaml` automáticamente.
4. En ~3 minutos tendrás una URL pública estilo `https://janus-siac.onrender.com`.

**Configurar dominio `janus.siac.ai`:**
- Compra `siac.ai` (Namecheap, GoDaddy, Cloudflare…).
- En Render → tu servicio → **Settings → Custom Domain** → añade `janus.siac.ai`.
- En tu DNS añade un registro `CNAME`:  
  `janus  CNAME  janus-siac.onrender.com`
- HTTPS se emite automáticamente (Let's Encrypt).

## Opción B — Railway.app

1. https://railway.app → **New project** → **Deploy from GitHub repo**.
2. Railway lee `railway.toml`. Listo.
3. **Settings → Networking → Custom Domain** → `janus.siac.ai` + CNAME.

## Opción C — Fly.io

```bash
iwr https://fly.io/install.ps1 -useb | iex
fly auth signup
fly launch          # detecta Dockerfile
fly deploy
fly certs add janus.siac.ai
```

## Opción D — Docker (cualquier VPS, Coolify, Dokploy…)

```bash
docker build -t janus-siac .
docker run -d -p 80:8000 --name janus --restart unless-stopped janus-siac
```

## Opción E — Hugging Face Spaces (gratis, sin tarjeta)

1. https://huggingface.co/new-space → **SDK: Docker**.
2. Sube los archivos del proyecto (drag&drop o git).
3. URL pública: `https://huggingface.co/spaces/TU_USER/janus-siac`.
4. Para dominio propio necesitas plan Pro de HF Spaces.

---

## Verificar el deploy

Una vez online, abre:
- `https://TU_DOMINIO/` → la UI completa
- `https://TU_DOMINIO/api/health` → debe responder `{"ok": true, ...}`

## Variables de entorno (opcionales)

| Variable | Default | Uso |
|---|---|---|
| `PORT` | 5000 | Puerto donde corre Flask (lo set Render/Railway/Fly automáticamente) |
| `FLASK_ENV` | `development` | Pon `production` para desactivar debug |
| `HOST` | `0.0.0.0` | Bind address |

## Nota sobre persistencia

El backend usa `STATE` global en memoria (single-user). En la nube cada despliegue arranca con estado vacío y, si reciben varios usuarios a la vez, comparten estado. Para producción multi-usuario:
- Usar `Flask-Session` con Redis (`pip install flask-session redis`), o
- Persistir el modelo entrenado a disco con `joblib.dump(model, 'model.pkl')` y cargar por sesión.

Para uso personal/demo está perfecto tal cual.
