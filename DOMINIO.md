# 🌐 Crear y conectar tu propio dominio web — JNUS AI

Guía paso a paso para que `jnus-ai.com` (o el nombre que elijas) apunte a tu app
desplegada en **Render**. No toca el backend ni los modelos: solo infraestructura.

---

## 1. Elige y registra el dominio

Sugerencias de nombre (verifica disponibilidad):

| Dominio                | Estilo            | Aprox. /año |
|------------------------|-------------------|-------------|
| `jnus-ai.com`          | Marca + IA        | $12         |
| `jnusfinance.com`      | Financiero        | $12         |
| `jnuscredito.com`      | Español / crédito | $12         |
| `jnus.app`             | Corto, moderno    | $18         |
| `jnus.ec`              | Ecuador 🇪🇨        | $25–40      |

**Dónde registrarlo** (cualquiera sirve):
- [Namecheap](https://www.namecheap.com) — barato y sencillo
- [Cloudflare Registrar](https://dash.cloudflare.com) — al costo, sin sobreprecio (recomendado)
- [GoDaddy](https://godaddy.com) — popular en LatAm

> 💡 Si quieres `.ec` para Ecuador, se registra en **NIC.ec**.

---

## 2. Despliega la app en Render (si aún no lo está)

El proyecto ya está listo para Render (`render.yaml`).

1. Sube el repo a GitHub.
2. En [render.com](https://render.com) → **New + → Blueprint** → selecciona el repo.
3. Render lee `render.yaml` y crea el servicio `janus-siac`.
4. En **Environment**, define las variables (Settings → Environment):
   - `JNUS_SECRET_KEY` = una cadena larga aleatoria
   - `JNUS_ADMIN_USER` = tu usuario admin
   - `JNUS_ADMIN_PASSWORD` = tu contraseña admin
5. Quedará una URL tipo `https://janus-siac.onrender.com`.

---

## 3. Conecta el dominio a Render

1. En tu servicio de Render → **Settings → Custom Domains → Add Custom Domain**.
2. Escribe tu dominio: `jnus-ai.com` y también `www.jnus-ai.com`.
3. Render te mostrará los **registros DNS** que debes crear.

### Registros DNS típicos (créalos en tu registrador)

| Tipo    | Nombre / Host | Valor                         |
|---------|---------------|-------------------------------|
| `A`     | `@`           | `216.24.57.1` (el que dé Render) |
| `CNAME` | `www`         | `janus-siac.onrender.com`     |

> Render te dará los valores exactos. Usa **esos**, no los del ejemplo.

4. Espera la propagación DNS (de 10 min a 24 h).
5. Render emite el certificado **HTTPS (SSL) gratis** automáticamente. ✅

---

## 4. Rutas finales de tu plataforma

Una vez conectado el dominio:

| URL                              | Quién entra        | Qué hace                          |
|----------------------------------|--------------------|-----------------------------------|
| `https://jnus-ai.com/`           | Público            | Redirige a la app                 |
| `https://jnus-ai.com/app`        | Clientes           | Evaluación crediticia (inferencia)|
| `https://jnus-ai.com/admin`      | Solo tú (login)    | Entrenamiento, métricas, cerebro IA|

---

## 5. (Opcional) Forzar `www` → raíz o viceversa

Para que `www.jnus-ai.com` y `jnus-ai.com` lleven al mismo sitio,
en Render agrega ambos dominios; Render gestiona la redirección.

---

## 6. Verifica

```bash
# Comprueba que el dominio resuelve y responde
curl -I https://jnus-ai.com/api/health
# Debe devolver HTTP/2 200
```

---

## Resumen rápido (TL;DR)

1. Registra `jnus-ai.com` en Cloudflare/Namecheap.
2. Despliega en Render con el `render.yaml` del repo.
3. Render → Custom Domains → agrega el dominio.
4. Copia los registros DNS de Render a tu registrador.
5. Espera, y listo: HTTPS automático. 🎉

> El reentrenamiento del modelo desde `/admin` **no requiere volver a desplegar**:
> el bundle se actualiza en caliente y `/app` usa siempre el más reciente.
