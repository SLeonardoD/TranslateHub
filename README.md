# 🌐 Servidor de Traducción Local

API HTTP local para servir capacidades de traducción usando modelos de Hugging Face con `transformers`. Diseñada para que otras aplicaciones de tu red (o de tu propio PC) la consuman como servicio centralizado, sin necesidad de implementar lógica de traducción en cada una.

## ✨ Características

- **Carga perezosa**: ningún modelo se carga en RAM hasta la primera petición.
- **Un solo modelo en memoria a la vez**: se libera el anterior antes de cargar otro.
- **Catálogo flexible**: varios modelos por par de idiomas (rápido, equilibrado, alta calidad).
- **Modelos multilingües**: soporte para NLLB con múltiples pares desde una sola descarga.
- **Descarga controlada**: la API **nunca** descarga modelos; eso se hace explícitamente desde el CLI.
- **Selección flexible**: el cliente puede pedir un modelo explícito o dejar que el servidor resuelva según idioma y tier.
- **Thread-safe**: un `threading.Lock` serializa carga + inferencia (operación atómica).
- **Detección de IPs de LAN**: `run.py` imprime las IPs locales sin depender de salida a Internet.

---

## 📁 Estructura del proyecto

```text
mi_servidor_traduccion/
├── app/
│   ├── __init__.py
│   ├── main.py              # Endpoints FastAPI
│   ├── config.py            # Rutas y carga del catálogo
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py    # Excepciones del dominio
│   │   └── model_manager.py # Gestor de modelos + CLI
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Esquemas Pydantic
│   └── services/
│       ├── __init__.py
│       └── translation.py   # Orquestación de la lógica de negocio
├── models/                  # Modelos descargados (auto-creado)
├── model_info.json          # Catálogo de modelos
├── pyproject.toml           # Configuración de ruff / pytest
├── requirements.txt         # Dependencias de producción
├── requirements-dev.txt     # Dependencias de desarrollo
├── run.py                   # Arranque con detección de IP
└── .gitignore
```

---

## 🧰 Requisitos

- **Python 3.11+**
- **pip** actualizado
- (Opcional) GPU NVIDIA con CUDA si quieres aceleración por hardware. La CPU es suficiente para OPUS-MT.

---

## 🚀 Puesta en marcha

### 1. Instalar dependencias

**Desarrollo** (recomendado para empezar):
```bash
pip install -r requirements-dev.txt
```

**Producción**:
```bash
pip install -r requirements.txt
```

### 2. Verificar el catálogo de modelos

```bash
python -m app.core.model_manager list
```

Salida esperada:

```text
  ⬇️ opus-mt-en-es                       [fast   ] OPUS-MT inglés→español, rápido y ligero
  ⬇️ opus-mt-es-en                       [fast   ] OPUS-MT español→inglés, rápido y ligero
  ⬇️ nllb-200-distilled-600M             [quality] NLLB-200 distilled 600M, multilingüe de alta calidad
```

### 3. Descargar los modelos que vayas a usar

```bash
# Un modelo específico
python -m app.core.model_manager download opus-mt-en-es

# Varios a la vez (debes repetir el comando por cada uno)
python -m app.core.model_manager download nllb-200-distilled-600M

# O descargar TODOS los del catálogo
python -m app.core.model_manager download-all
```

> ⚠️ La API **no descarga modelos**. Si una petición llega para un modelo no descargado, devolverá un error 404 con el comando exacto a ejecutar.

### 4. Arrancar el servidor

```bash
python run.py
```

Salida esperada:

```text
============================================================
🚀 Servidor de traducción iniciado
   📍 Localhost:  http://127.0.0.1:8000
   🌐 Red local:  http://192.168.1.42:8000
   📖 Docs:       http://192.168.1.42:8000/docs
============================================================
```

Apunta la URL de la red local: es la que usarán tus otras aplicaciones.

---

## 🛠️ CLI: gestor de modelos

Todas las operaciones sobre modelos descargados se hacen desde el CLI.

```bash
python -m app.core.model_manager <comando> [args]
```

| Comando | Descripción |
|---|---|
| `list` | Lista todos los modelos del catálogo con su estado (descargado ✅ / no descargado ⬇️). |
| `pairs` | Muestra los pares de idiomas cubiertos y qué modelos los sirven. |
| `status` | Indica qué modelo está cargado en RAM y el estado de descargas. |
| `download <model_id>` | Descarga un modelo del catálogo a `models/`. |
| `download-all` | Descarga todos los modelos del catálogo. |

---

## 📚 Documentación de la API

Base URL: `http://<ip>:8000`

Documentación interactiva (Swagger): `http://<ip>:8000/docs`

### `GET /`

Estado del servidor.

**Respuesta** `200 OK`:
```json
{ "status": "ok" }
```

---

### `GET /models`

Lista todos los modelos del catálogo con su estado.

**Respuesta** `200 OK`:
```json
[
  {
    "model_id": "opus-mt-en-es",
    "hf_name": "Helsinki-NLP/opus-mt-en-es",
    "family": "opus-mt",
    "type": "pair",
    "tier": "fast",
    "description": "OPUS-MT inglés→español, rápido y ligero",
    "source_lang": "en",
    "target_lang": "es",
    "supported_pairs": null,
    "downloaded": true,
    "loaded": true
  },
  {
    "model_id": "nllb-200-distilled-600M",
    "hf_name": "facebook/nllb-200-distilled-600M",
    "family": "nllb",
    "type": "multilingual",
    "tier": "quality",
    "description": "NLLB-200 distilled 600M, multilingüe de alta calidad",
    "source_lang": null,
    "target_lang": null,
    "supported_pairs": [
      ["en", "es"], ["es", "en"],
      ["en", "fr"], ["fr", "en"],
      ["en", "de"], ["de", "en"]
    ],
    "downloaded": true,
    "loaded": false
  }
]
```

- `downloaded`: si el modelo existe en `models/`.
- `loaded`: si actualmente está en RAM.

---

### `GET /pairs`

Lista los pares de idiomas cubiertos y qué modelos los sirven.

**Respuesta** `200 OK`:
```json
[
  { "source_lang": "en", "target_lang": "de", "models": ["nllb-200-distilled-600M"] },
  { "source_lang": "en", "target_lang": "es", "models": ["nllb-200-distilled-600M", "opus-mt-en-es"] },
  { "source_lang": "en", "target_lang": "fr", "models": ["nllb-200-distilled-600M"] },
  { "source_lang": "es", "target_lang": "en", "models": ["nllb-200-distilled-600M", "opus-mt-es-en"] }
]
```

---

### `POST /translate`

Traduce texto. Acepta dos formas de seleccionar el modelo:

- **Opción A**: `model_id` explícito.
- **Opción B**: `source_lang` + `target_lang` (+ `tier` opcional), y el servidor resuelve el mejor modelo.

**Body** (`application/json`):

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `text` | `string` | ✅ | Texto a traducir. |
| `model_id` | `string` | ❌* | ID del modelo del catálogo. |
| `source_lang` | `string` | ❌* | Código ISO 639-1 de origen (`en`, `es`, `fr`, `de`…). |
| `target_lang` | `string` | ❌* | Código ISO 639-1 de destino. |
| `tier` | `string` | ❌ | `fast`, `balanced`, `quality`. Solo si se usa la Opción B. |
| `max_length` | `int` | ❌ | Longitud máxima de la traducción. Si se omite, usa `default_max_length` del modelo. |

\* Debes enviar `model_id` **o** (`source_lang` + `target_lang`). Si envías `model_id`, `source_lang`/`target_lang` son opcionales salvo para modelos multilingües (donde son obligatorios).

**Respuesta** `200 OK`:
```json
{
  "translated_text": "Hola mundo",
  "model_id": "opus-mt-en-es",
  "source_lang": "en",
  "target_lang": "es"
}
```

**Códigos de error**:

| Código | Causa |
|---|---|
| `404` | Modelo no definido en el catálogo, no descargado, o par de idiomas no soportado. |
| `422` | Body inválido (falta `text`, ni `model_id` ni par de idiomas…). |
| `500` | Error al cargar el modelo en memoria o durante la inferencia. |

---

## 💡 Ejemplos de uso

### Ejemplo 1 — Traducción básica (auto-resolver por par)

Si solo indicas el par, el servidor elige el modelo `fast` por defecto (OPUS-MT).

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_lang": "en", "target_lang": "es"}'
```

Respuesta:
```json
{
  "translated_text": "Hola mundo",
  "model_id": "opus-mt-en-es",
  "source_lang": "en",
  "target_lang": "es"
}
```

---

### Ejemplo 2 — Forzar calidad (tier `quality`)

Fuerza el uso de un modelo de mayor calidad (en este catálogo, NLLB).

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog", "source_lang": "en", "target_lang": "es", "tier": "quality"}'
```

Respuesta:
```json
{
  "translated_text": "El rápido zorro marrón salta sobre el perro perezoso",
  "model_id": "nllb-200-distilled-600M",
  "source_lang": "en",
  "target_lang": "es"
}
```

> 🔄 Al cambiar de modelo, el servidor libera el anterior de RAM antes de cargar el nuevo. La petición esperará a que termine la operación en curso, si hay alguna.

---

### Ejemplo 3 — Modelo explícito con idioma destino distinto

Usa NLLB (multilingüe) para traducir a un idioma que ningún modelo de par cubre.

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model_id": "nllb-200-distilled-600M", "source_lang": "en", "target_lang": "fr"}'
```

Respuesta:
```json
{
  "translated_text": "Bonjour le monde",
  "model_id": "nllb-200-distilled-600M",
  "source_lang": "en",
  "target_lang": "fr"
}
```

> ℹ️ Para modelos multilingües, `source_lang` y `target_lang` son **obligatorios**. Para modelos de par, si los envías deben coincidir con el par nativo del modelo.

---

### Ejemplo 4 — Ver pares disponibles

Antes de enviar una petición, consulta qué idiomas puedes traducir y con qué modelos.

```bash
curl http://localhost:8000/pairs
```

Respuesta:
```json
[
  { "source_lang": "en", "target_lang": "de", "models": ["nllb-200-distilled-600M"] },
  { "source_lang": "en", "target_lang": "es", "models": ["nllb-200-distilled-600M", "opus-mt-en-es"] },
  { "source_lang": "en", "target_lang": "fr", "models": ["nllb-200-distilled-600M"] },
  { "source_lang": "es", "target_lang": "en", "models": ["nllb-200-distilled-600M", "opus-mt-es-en"] }
]
```

---

### Ejemplo 5 — Estado de los modelos

```bash
curl http://localhost:8000/models
```

Te dice qué modelos están descargados (`downloaded: true`) y cuál está en RAM ahora mismo (`loaded: true`). Útil para depurar antes de enviar peticiones.

---

### Ejemplo 6 — Uso desde Python

```python
import requests

URL = "http://192.168.1.42:8000/translate"

def traducir(texto: str, src: str, tgt: str, tier: str | None = None) -> str:
    payload = {"text": texto, "source_lang": src, "target_lang": tgt}
    if tier:
        payload["tier"] = tier
    r = requests.post(URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["translated_text"]

print(traducir("Hello world", "en", "es"))                    # rápido (OPUS-MT)
print(traducir("Hello world", "en", "es", tier="quality"))    # NLLB
print(traducir("Hello world", "en", "fr"))                    # NLLB (único disponible)
```

---

## 🗂️ Catálogo de modelos (`model_info.json`)

El catálogo define qué modelos conoce el sistema. Puedes tener **muchos definidos** sin estar descargados. La API solo los operará cuando estén descargados.

### Formato de un modelo de par (`pair`)

```json
"opus-mt-en-es": {
  "hf_name": "Helsinki-NLP/opus-mt-en-es",
  "family": "opus-mt",
  "type": "pair",
  "source_lang": "en",
  "target_lang": "es",
  "tier": "fast",
  "description": "OPUS-MT inglés→español, rápido y ligero",
  "default_max_length": 512
}
```

### Formato de un modelo multilingüe (`multilingual`)

```json
"nllb-200-distilled-600M": {
  "hf_name": "facebook/nllb-200-distilled-600M",
  "family": "nllb",
  "type": "multilingual",
  "tier": "quality",
  "supported_pairs": [["en", "es"], ["en", "fr"], ["en", "de"]],
  "language_code_map": {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn"
  },
  "description": "NLLB-200 distilled 600M, multilingüe",
  "default_max_length": 512
}
```

### Campos

| Campo | Aplica a | Descripción |
|---|---|---|
| `hf_name` | Todos | ID del modelo en Hugging Face. Define la ruta local (`models/<hf_name con __ >`). |
| `family` | Todos | Estrategia de inferencia. Soportados: `opus-mt` (genérico) y `nllb`. Extensible. |
| `type` | Todos | `pair` o `multilingual`. |
| `tier` | Todos | `fast`, `balanced` o `quality`. Se usa en la auto-resolución. |
| `source_lang` / `target_lang` | Solo `pair` | Par nativo del modelo. |
| `supported_pairs` | Solo `multilingual` | Lista de pares `[src, tgt]` que cubre. |
| `language_code_map` | Solo `multilingual` | Códigos ISO cortos → códigos internos del tokenizador. |
| `default_max_length` | Todos | Longitud máxima por defecto de la traducción. |
| `description` | Todos | Texto libre para humanos. |

### Añadir un modelo nuevo

1. Añade su entrada al `model_info.json`.
2. Descárgalo con `python -m app.core.model_manager download <model_id>`.
3. Reinicia el servidor (o espera a la siguiente petición si es el primero tras el arranque).

---

## 🏛️ Arquitectura

```text
┌──────────────────────────────────────────────────────────┐
│  Cliente (curl / app Python / navegador)                 │
└────────────────────────────┬─────────────────────────────┘
                             │ HTTP
                             ▼
┌──────────────────────────────────────────────────────────┐
│  app/main.py           →  Capa HTTP (FastAPI)            │
│  app/models/schemas.py →  Validación Pydantic            │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  app/services/translation.py  →  Lógica de negocio       │
│    · Resuelve el modelo adecuado                         │
│    · Valida coherencia par/modelo                        │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  app/core/model_manager.py  →  Ciclo de vida del modelo  │
│    · Descarga (solo CLI)                                 │
│    · Carga en RAM (lazy, uno a la vez)                   │
│    · Inferencia (con threading.Lock)                     │
└──────────────────────────────────────────────────────────┘
```

**Reglas de oro**:

- **La API nunca descarga**. Si el modelo no está local, se devuelve un error 404 accionable.
- **Un solo modelo en RAM**. Cargar otro descarga el anterior.
- **Acceso atómico**. El lock cubre carga + inferencia: si llega una petición para otro modelo mientras hay una en curso, espera a que termine.

---

## 🩺 Notas y solución de problemas

### La API devuelve 404 "no está descargado localmente"

Es el comportamiento esperado. Ejecuta el comando que indica el error:

```bash
python -m app.core.model_manager download <model_id>
```

### Windows pregunta por el firewall al arrancar

Acepta el permiso en **redes privadas**. Si lo rechazas, otros equipos de la LAN no podrán acceder.

### Quiero fijar la IP del servidor

Configura una **reserva DHCP** para tu PC en el router. Así la URL `http://<ip>:8000` no cambiará entre reinicios.

### Qué modelos elegir

- **OPUS-MT** (`Helsinki-NLP/opus-mt-*`): muy rápidos, ideales para traducción en tiempo real. Cada par requiere su propio modelo.
- **NLLB-200-distilled-600M**: un solo modelo cubre 200 idiomas. Más lento que OPUS-MT pero de mayor calidad y más flexible.

Ambos conviven perfectamente en el catálogo. Usa `tier` para elegir según contexto.

### Cambiar el puerto o la IP de escucha

Edita las constantes al inicio de `run.py`:

```python
HOST = "0.0.0.0"   # todas las interfaces
PORT = 8000
```

### Parar el servidor

`Ctrl+C` en la terminal donde corre `run.py`.

---

## 📜 Licencia

Proyecto interno. Los modelos descargados desde Hugging Face mantienen sus licencias originales (consulta la ficha de cada modelo en el Hub).