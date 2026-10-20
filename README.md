# 🌐 Servidor de Traducción Local

API HTTP local para servir capacidades de traducción usando modelos de Hugging Face con `transformers`. Diseñada para que otras aplicaciones de tu red (o de tu propio PC) la consuman como servicio centralizado, sin necesidad de implementar lógica de traducción en cada una.

## ✨ Características

- **Carga perezosa**: ningún modelo se carga en RAM hasta la primera petición (o al arrancar, con `TRANSLATOR_PRELOAD_MODEL`).
- **Un modelo en memoria por defecto**: se libera el anterior antes de cargar otro. Si te sobra RAM, `TRANSLATOR_MAX_LOADED_MODELS` conserva los últimos usados (LRU).
- **Cola FIFO**: las traducciones se atienden **de una en una, en orden de llegada**, sin bloquear el servidor: `/`, `/models` y `/pairs` responden siempre, incluso con una traducción en curso. Si la cola se llena, responde `503`.
- **Textos largos sin pérdidas**: se trocean por oraciones, se traducen por lotes y se reensamblan conservando saltos de línea. Nada se corta en silencio; si algo puede estar incompleto, la respuesta lo indica (`truncated`, `warnings`).
- **Traducción por lotes**: `POST /translate/batch` traduce varios textos en una sola petición.
- **Catálogo flexible y validado**: varios modelos por par de idiomas (rápido, equilibrado, alta calidad). Si `model_info.json` tiene un error, el servidor se niega a arrancar y dice exactamente cuál.
- **Modelos multilingües**: soporte para NLLB con múltiples pares desde una sola descarga.
- **Descarga controlada y verificada**: la API **nunca** descarga modelos; eso se hace desde el CLI. Un modelo solo cuenta como descargado cuando la descarga terminó (una descarga interrumpida ya no se confunde con una completa) y solo se bajan los pesos PyTorch necesarios.
- **Selección flexible**: el cliente puede pedir un modelo explícito o dejar que el servidor resuelva según idioma y tier.
- **Errores claros**: `404` (modelo/par no disponible), `422` (petición inválida), `503` (cola llena), `500` (fallo real de carga o inferencia).
- **Seguridad opcional**: API key por cabecera y CORS, desactivados por defecto.
- **Detección de IPs de LAN**: `run.py` imprime las IPs locales sin depender de Internet, descartando adaptadores virtuales (Docker, WSL, VirtualBox, VPN…) y poniendo primero la interfaz real.

---

## 📁 Estructura del proyecto

```text
mi_servidor_traduccion/
├── app/
│   ├── __init__.py
│   ├── main.py              # Endpoints FastAPI, errores HTTP, API key y CORS
│   ├── config.py            # Ajustes (variables de entorno) y catálogo validado
│   ├── logging_setup.py     # Configuración de logs
│   ├── core/
│   │   ├── __init__.py
│   │   ├── concurrency.py   # Cola FIFO de traducciones
│   │   ├── exceptions.py    # Excepciones del dominio
│   │   ├── languages.py     # Normalización de códigos de idioma
│   │   ├── model_manager.py # Gestor de modelos + CLI
│   │   └── text_utils.py    # Troceado de textos largos
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Esquemas Pydantic
│   └── services/
│       ├── __init__.py
│       └── translation.py   # Orquestación de la lógica de negocio
├── tests/                   # Suite pytest
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
- (Opcional) GPU NVIDIA con CUDA si quieres aceleración por hardware (se usa `float16` automáticamente). La CPU es suficiente para OPUS-MT.

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
  ⬇️  opus-mt-en-es                       [fast   ] OPUS-MT inglés→español, rápido y ligero
  ⬇️  opus-mt-es-en                       [fast   ] OPUS-MT español→inglés, rápido y ligero
  ⬇️  nllb-200-distilled-600M             [quality] NLLB-200 distilled 600M, multilingüe de alta calidad
```

Leyenda: ✅ descargado · ⬇️ no descargado · ⚠️ descarga incompleta (o hecha con una versión anterior).

### 3. Descargar los modelos que vayas a usar

```bash
# Un modelo específico
python -m app.core.model_manager download opus-mt-en-es

# O descargar TODOS los del catálogo
python -m app.core.model_manager download-all

# Volver a validar/descargar un modelo ya descargado
python -m app.core.model_manager download opus-mt-en-es --force
```

> ⚠️ La API **no descarga modelos**. Si una petición llega para un modelo no descargado, devolverá un error 404 con el comando exacto a ejecutar.

> 🔁 **Si ya tenías modelos descargados con una versión anterior**, aparecerán como ⚠️ porque aún no tienen la marca de "descarga completa". Ejecuta `download <model_id>` una vez por cada uno: reutiliza/valida los archivos existentes, descarga lo que falte y lo marca como completo. Puedes borrar de `models/` los pesos duplicados (`*.h5`, `*.ot`…) para recuperar espacio.

### 4. Arrancar el servidor

```bash
python run.py
```

Salida esperada:

```text
============================================================
🚀 Servidor de traducción iniciado
   📍 Localhost:  http://127.0.0.1:8000
   🌐 Red local: http://192.168.1.42:8000   (Wi-Fi)
   📖 Docs:       http://192.168.1.42:8000/docs
   📦 Modelos en el catálogo: 3
============================================================
```

Apunta la URL de **Red local**: es la que usarán tus otras aplicaciones. Si el equipo tiene varias redes reales, se listan también como "Otra red", con la interfaz principal siempre primero. Los adaptadores virtuales (Docker, WSL, VirtualBox, VPN…) no se listan.

---

## ⚙️ Configuración (variables de entorno)

Todas son opcionales. Se leen al arrancar.

| Variable | Por defecto | Descripción |
|---|---|---|
| `TRANSLATOR_HOST` | `0.0.0.0` | Interfaz de escucha. Usa `127.0.0.1` para aceptar solo conexiones locales. |
| `TRANSLATOR_PORT` | `8000` | Puerto. |
| `TRANSLATOR_PRELOAD_MODEL` | — | `model_id` a cargar al arrancar, para que la primera petición no espere. |
| `TRANSLATOR_MAX_LOADED_MODELS` | `1` | Modelos simultáneos en RAM (se descarta el menos usado). OPUS-MT pesa ~300 MB; NLLB ~2,5 GB. |
| `TRANSLATOR_MAX_PENDING_REQUESTS` | `16` | Peticiones de traducción en curso + en cola. Al superarlo → `503`. |
| `TRANSLATOR_DEVICE` | `auto` | `auto`, `cpu` o `cuda`. |
| `TRANSLATOR_CHUNK_TARGET_TOKENS` | `200` | Tamaño objetivo de cada fragmento al trocear textos largos (mínimo 16). |
| `TRANSLATOR_BATCH_SIZE` | `8` | Fragmentos traducidos a la vez. Bájalo si te quedas sin RAM. |
| `TRANSLATOR_MAX_TEXT_CHARS` | `20000` | Caracteres máximos por texto. |
| `TRANSLATOR_MAX_BATCH_ITEMS` | `32` | Textos máximos en `/translate/batch`. |
| `TRANSLATOR_MAX_BATCH_CHARS` | `100000` | Caracteres totales máximos en `/translate/batch`. |
| `TRANSLATOR_API_KEY` | — | Si se define, `/models`, `/pairs` y `/translate*` exigen la cabecera `X-API-Key`. |
| `TRANSLATOR_CORS_ORIGINS` | — | Orígenes permitidos para navegadores, separados por comas. |
| `TRANSLATOR_MODELS_DIR` | `./models` | Carpeta de los modelos descargados. |
| `TRANSLATOR_LOG_LEVEL` | `INFO` | Nivel de logs. |

Ejemplos:

```bash
# Linux / macOS
TRANSLATOR_PRELOAD_MODEL=opus-mt-en-es TRANSLATOR_API_KEY=mi-clave python run.py

# Windows (PowerShell)
$env:TRANSLATOR_PRELOAD_MODEL = "opus-mt-en-es"; python run.py
```

> Con `TRANSLATOR_PRELOAD_MODEL` el banner con las URLs aparece antes de que termine la carga; el servidor empieza a aceptar peticiones cuando el modelo está listo.

---

## 🛠️ CLI: gestor de modelos

Todas las operaciones sobre modelos descargados se hacen desde el CLI.

```bash
python -m app.core.model_manager <comando> [args]
```

| Comando | Descripción |
|---|---|
| `list` | Lista todos los modelos del catálogo con su estado de descarga. |
| `pairs` | Muestra los pares de idiomas cubiertos y qué modelos los sirven. |
| `status` | Pregunta al **servidor en marcha** qué modelos tiene cargados en RAM y muestra el estado de descargas. Si el servidor no está corriendo, solo muestra las descargas. |
| `download <model_id> [--force]` | Descarga un modelo del catálogo a `models/`. Con `--force` vuelve a validarlo/descargarlo. |
| `download-all [--force]` | Descarga todos los modelos del catálogo. |

---

## 📚 Documentación de la API

Base URL: `http://<ip>:8000`

Documentación interactiva (Swagger): `http://<ip>:8000/docs`

Si definiste `TRANSLATOR_API_KEY`, añade la cabecera `X-API-Key: <tu clave>` a todas las peticiones salvo `GET /`. Sin clave válida → `401`.

### `GET /`

Estado del servidor (siempre abierto, útil como health-check).

**Respuesta** `200 OK`:
```json
{
  "status": "ok",
  "loaded_models": ["opus-mt-en-es"],
  "queue": { "pending": 0, "max": 16 }
}
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

- `downloaded`: la descarga terminó y verificó correctamente.
- `loaded`: el modelo está actualmente en RAM.

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

Traduce un texto. Acepta dos formas de seleccionar el modelo:

- **Opción A**: `model_id` explícito.
- **Opción B**: `source_lang` + `target_lang` (+ `tier` opcional), y el servidor resuelve el mejor modelo.

**Body** (`application/json`):

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `text` | `string` | ✅ | Texto a traducir (no vacío; máx. `TRANSLATOR_MAX_TEXT_CHARS`). |
| `model_id` | `string` | ❌* | ID del modelo del catálogo. |
| `source_lang` | `string` | ❌* | Código ISO 639-1 de origen (`en`, `es`, `fr`, `de`…). Se normaliza: `"EN"` y `"es-ES"` valen como `"en"` y `"es"`. |
| `target_lang` | `string` | ❌* | Código ISO 639-1 de destino. |
| `tier` | `string` | ❌ | `fast`, `balanced`, `quality`. Solo si se usa la Opción B. |
| `max_length` | `int` | ❌ | Máximo de **tokens generados por fragmento** (1–1024). Si se omite, usa `default_max_length` del modelo. |

\* Debes enviar `model_id` **o** (`source_lang` + `target_lang`). Si envías `model_id`, `source_lang`/`target_lang` son opcionales salvo para modelos multilingües (donde son obligatorios). En modelos de par, si los envías **deben coincidir con el par nativo del modelo** (tanto origen como destino).

**Respuesta** `200 OK`:
```json
{
  "translated_text": "Hola mundo",
  "chunks": 1,
  "truncated": false,
  "warnings": [],
  "model_id": "opus-mt-en-es",
  "source_lang": "en",
  "target_lang": "es"
}
```

| Campo | Descripción |
|---|---|
| `chunks` | En cuántos fragmentos se dividió el texto para traducirlo (1 = sin dividir). |
| `truncated` | `true` si alguna traducción agotó `max_length` y **puede estar incompleta**. |
| `warnings` | Avisos legibles: truncado, oraciones muy largas partidas por palabras… Vacío si todo fue bien. |

**Códigos de error** (el cuerpo es `{"detail": ...}`: un texto, o una lista con los campos inválidos en los `422` de validación):

| Código | Causa |
|---|---|
| `401` | Falta o es incorrecta la API key (solo si `TRANSLATOR_API_KEY` está definida). |
| `404` | Modelo no definido en el catálogo, no descargado (o descarga incompleta), o par de idiomas no soportado. |
| `422` | Petición inválida: texto vacío o demasiado largo, falta `model_id` o par de idiomas, `tier`/`max_length` fuera de rango, idioma no soportado por el modelo, modelo multilingüe sin idiomas… |
| `503` | Cola llena (`TRANSLATOR_MAX_PENDING_REQUESTS`). Incluye cabecera `Retry-After`. |
| `500` | Fallo real al cargar el modelo en memoria o durante la inferencia (por ejemplo, falta de memoria). El traceback queda en la consola del servidor. |

---

### `POST /translate/batch`

Traduce varios textos con **el mismo modelo** en una sola petición. Misma selección de modelo y mismos campos que `/translate`, pero con `texts` (lista) en lugar de `text`.

```bash
curl -X POST http://localhost:8000/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Good morning"], "source_lang": "en", "target_lang": "es"}'
```

**Respuesta** `200 OK`:
```json
{
  "model_id": "opus-mt-en-es",
  "source_lang": "en",
  "target_lang": "es",
  "results": [
    { "translated_text": "Hola mundo", "chunks": 1, "truncated": false, "warnings": [] },
    { "translated_text": "Buenos días", "chunks": 1, "truncated": false, "warnings": [] }
  ]
}
```

`results` conserva el orden de `texts`. Límites: `TRANSLATOR_MAX_BATCH_ITEMS` textos y `TRANSLATOR_MAX_BATCH_CHARS` caracteres en total.

---

## 📏 Textos largos: cómo se manejan

Los modelos trabajan mejor frase a frase y tienen un límite de entrada (~512 tokens). Antes, lo que pasaba del límite se **descartaba sin avisar**. Ahora:

1. **Un texto corto no cambia**: si cabe en un fragmento (≈200 tokens por defecto), se traduce de una vez, igual que siempre.
2. **Un texto largo se parte por oraciones** y se agrupan hasta `TRANSLATOR_CHUNK_TARGET_TOKENS`. Cada fragmento se traduce por separado (en lotes) y se reensambla en orden.
3. **Los saltos de línea se conservan**: un fragmento nunca cruza un salto de línea, así que párrafos y listas mantienen su estructura.
4. **Una oración gigante** (más larga que un fragmento) se divide por palabras y se añade un aviso en `warnings`.
5. **Si una traducción agota `max_length`**, `truncated` pasa a `true` y se añade un aviso. Nunca falla en silencio.

Si tu aplicación prefiere controlar el troceado (por ejemplo, por párrafos), puedes enviar los fragmentos ya separados con `/translate/batch`: cada texto del lote se trocea de forma independiente solo si es largo.

---

## 🚦 Cola de peticiones

El servidor atiende las traducciones **de una en una, en orden de llegada**, pensado para equipos modestos:

- Mientras una traducción está en curso, las demás esperan su turno sin consumir CPU.
- `GET /`, `/models` y `/pairs` **no esperan** a la cola: responden al instante.
- Si hay `TRANSLATOR_MAX_PENDING_REQUESTS` peticiones pendientes (en curso + en cola), las nuevas reciben `503` con `Retry-After: 5`.
- `GET /` muestra cuántas hay pendientes (`queue.pending`).

> ⏱️ Con cola, un cliente puede esperar mucho: usa un *timeout* generoso (120 s o más) o consulta `GET /` antes de enviar.

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
  "chunks": 1,
  "truncated": false,
  "warnings": [],
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

> 🔄 Al cambiar de modelo, el servidor libera el anterior de RAM antes de cargar el nuevo (con `TRANSLATOR_MAX_LOADED_MODELS=1`). La petición esperará a que termine la operación en curso, si hay alguna.

---

### Ejemplo 3 — Modelo explícito con idioma destino distinto

Usa NLLB (multilingüe) para traducir a un idioma que ningún modelo de par cubre.

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model_id": "nllb-200-distilled-600M", "source_lang": "en", "target_lang": "fr"}'
```

> ℹ️ Para modelos multilingües, `source_lang` y `target_lang` son **obligatorios** (sin ellos: `422`). Para modelos de par, si los envías deben coincidir con el par nativo: pedir `opus-mt-en-es` con `target_lang: "fr"` devuelve `404` en vez de traducir a otro idioma.

---

### Ejemplo 4 — Ver pares disponibles

```bash
curl http://localhost:8000/pairs
```

---

### Ejemplo 5 — Estado de los modelos

```bash
curl http://localhost:8000/models
```

Te dice qué modelos están descargados (`downloaded: true`) y cuáles están en RAM ahora mismo (`loaded: true`). Útil para depurar antes de enviar peticiones.

---

### Ejemplo 6 — Uso desde Python

```python
import requests

BASE = "http://192.168.1.42:8000"
HEADERS = {}  # {"X-API-Key": "mi-clave"} si activaste TRANSLATOR_API_KEY


def traducir(texto: str, src: str, tgt: str, tier: str | None = None) -> str:
    payload = {"text": texto, "source_lang": src, "target_lang": tgt}
    if tier:
        payload["tier"] = tier
    r = requests.post(f"{BASE}/translate", json=payload, headers=HEADERS, timeout=120)
    r.raise_for_status()
    data = r.json()
    for aviso in data["warnings"]:
        print("⚠️", aviso)  # p. ej. traducción posiblemente incompleta
    return data["translated_text"]


print(traducir("Hello world", "en", "es"))  # rápido (OPUS-MT)
print(traducir("Hello world", "en", "es", tier="quality"))  # NLLB
print(traducir("Hello world", "en", "fr"))  # NLLB (único disponible)
```

---

## 🗂️ Catálogo de modelos (`model_info.json`)

El catálogo define qué modelos conoce el sistema. Puedes tener **muchos definidos** sin estar descargados. La API solo los operará cuando estén descargados.

**Se valida al arrancar** (y en cada comando del CLI): campos obligatorios, valores de `type`/`tier`, coherencia del par, `language_code_map` completo y campos desconocidos (para cazar erratas). Todos los errores se muestran juntos:

```text
❌ Catálogo model_info.json inválido:
  - opus-mt-en-es.tier: Input should be 'fast', 'balanced' or 'quality'
  - mi-modelo.hf_name: Field required
```

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
| `hf_name` | Todos | ID del modelo en Hugging Face (`dueño/nombre`). Define la ruta local (`models/<hf_name con __ >`). |
| `family` | Todos | Estrategia de inferencia. `opus-mt` (genérico, por defecto) y `nllb` (obligatorio para `multilingual`). |
| `type` | Todos | `pair` o `multilingual`. |
| `tier` | Todos | `fast`, `balanced` (por defecto) o `quality`. Se usa en la auto-resolución. |
| `source_lang` / `target_lang` | Solo `pair` | Par nativo del modelo (códigos de 2-3 letras en minúscula). |
| `supported_pairs` | Solo `multilingual` | Lista de pares `[src, tgt]` que cubre. |
| `language_code_map` | Solo `multilingual` | Códigos ISO cortos → códigos internos del tokenizador. Debe incluir todos los idiomas de `supported_pairs`. |
| `default_max_length` | Todos | Máximo de tokens generados por fragmento (por defecto 512). |
| `description` | Todos | Texto libre para humanos. |

### Añadir un modelo nuevo

1. Añade su entrada al `model_info.json`.
2. Descárgalo con `python -m app.core.model_manager download <model_id>`.
3. Reinicia el servidor.

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
│  app/core/concurrency  →  Cola FIFO (sin bloquear el loop)│
└────────────────────────────┬─────────────────────────────┘
                             │ (un hilo, de uno en uno)
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
│    · Descarga verificada (solo CLI)                      │
│    · Carga en RAM (lazy, LRU)                            │
│    · Troceado (text_utils) + inferencia por lotes        │
└──────────────────────────────────────────────────────────┘
```

**Reglas de oro**:

- **La API nunca descarga**. Si el modelo no está local, se devuelve un error 404 accionable.
- **Pocos modelos en RAM** (uno por defecto). Cargar otro descarta el menos usado.
- **Acceso atómico**. Cola FIFO + lock: si llega una petición para otro modelo mientras hay una en curso, espera su turno.
- **Nada se pierde en silencio**. Texto largo → troceado; salida cortada → `truncated`.

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

La suite no necesita descargar modelos: usa un tokenizador y un modelo falsos. Incluye una prueba de integración con un modelo Marian diminuto (pesos aleatorios, sin red) que se ejecuta solo si tienes `torch`, `transformers` y `sentencepiece` instalados.

---

## 🩺 Notas y solución de problemas

### La API devuelve 404 "no está descargado localmente"

Es el comportamiento esperado. Ejecuta el comando que indica el error:

```bash
python -m app.core.model_manager download <model_id>
```

Si el mensaje dice "(o la descarga quedó incompleta)", es que la descarga se interrumpió o es de una versión anterior: el mismo comando la repara.

### El servidor responde 503

Hay demasiadas traducciones en cola. Reintenta (la respuesta trae `Retry-After`) o sube `TRANSLATOR_MAX_PENDING_REQUESTS`.

### La primera petición tarda mucho

Es la carga del modelo en RAM. Usa `TRANSLATOR_PRELOAD_MODEL=<model_id>` para cargarlo al arrancar.

### Windows pregunta por el firewall al arrancar

Acepta el permiso en **redes privadas**. Si lo rechazas, otros equipos de la LAN no podrán acceder.

### Quiero fijar la IP del servidor

Configura una **reserva DHCP** para tu PC en el router. Así la URL `http://<ip>:8000` no cambiará entre reinicios.

### `run.py` no muestra la IP que esperaba

Se muestran las IPs privadas de las interfaces activas, excluyendo adaptadores virtuales por su nombre (Docker, WSL/Hyper-V, VirtualBox, VMware, VPN…). La interfaz que el sistema usa para salir a la red va siempre primera. Si usas una VPN que enruta todo el tráfico, la IP real de tu LAN aparecerá como "Otra red".

### Qué modelos elegir

- **OPUS-MT** (`Helsinki-NLP/opus-mt-*`): muy rápidos, ideales para traducción en tiempo real. Cada par requiere su propio modelo.
- **NLLB-200-distilled-600M**: un solo modelo cubre 200 idiomas. Más lento que OPUS-MT pero de mayor calidad y más flexible.

Ambos conviven perfectamente en el catálogo. Usa `tier` para elegir según contexto.

### Cambiar el puerto o la IP de escucha

Usa las variables de entorno `TRANSLATOR_PORT` y `TRANSLATOR_HOST` (ver [Configuración](#️-configuración-variables-de-entorno)).

### Seguridad

Por defecto el servidor escucha en `0.0.0.0` **sin autenticación**: cualquier equipo de tu red puede usarlo. Para uso interno suele bastar; si no, define `TRANSLATOR_API_KEY`, o limita la escucha con `TRANSLATOR_HOST=127.0.0.1`. Si lo consume una página web, añade su origen en `TRANSLATOR_CORS_ORIGINS`.

### Parar el servidor

`Ctrl+C` en la terminal donde corre `run.py`.

---

## 📜 Licencia

Proyecto interno. Los modelos descargados desde Hugging Face mantienen sus licencias originales (consulta la ficha de cada modelo en el Hub).
