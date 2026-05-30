# RAG Go — Sistema RAG local con Go + Python + Qwen + Qdrant

Sistema RAG production-grade para documentos locales.  
Optimizado para hardware de consumo (I7-1255U / 16GB RAM).

## Stack

| Componente | Tecnología |
|-----------|-----------|
| Backend principal | Go 1.22 |
| Chunking / PDF | Python 3.11 + PyMuPDF |
| Embeddings | nomic-embed-text (Ollama) |
| LLM | qwen2.5:7b-instruct (Ollama) |
| Vector DB | Qdrant |
| Cache embeddings | bbolt (embebido en Go) |

## Estructura del proyecto

```
rag-go/
├── cmd/
│   ├── index/          # CLI para indexar PDFs
│   └── query/          # CLI interactivo para preguntas
├── internal/
│   ├── cache/          # Cache de embeddings (bbolt)
│   ├── chunker/        # Cliente HTTP para Python service
│   ├── config/         # Configuración centralizada
│   ├── embed/          # Cliente Ollama para embeddings
│   ├── llm/            # Cliente Ollama con streaming
│   └── store/          # Cliente Qdrant (gRPC)
├── chunker-service/    # Microservicio Python
│   ├── extractors/     # Extracción de texto (PyMuPDF)
│   ├── chunkers/       # Estrategias de chunking
│   ├── main.py
│   └── requirements.txt
└── data/
    ├── cache/          # Embeddings cacheados (bbolt)
    └── pdfs/           # Tus PDFs
```

## Setup

### 1. Ollama + modelos

```bash
# Instalar Ollama desde https://ollama.ai

# Bajar los modelos necesarios
ollama pull nomic-embed-text    # ~270MB - embeddings
ollama pull qwen2.5:7b-instruct # ~4.5GB - LLM de chat
```

### 2. Qdrant (Docker)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verificar que funcione
curl http://localhost:6333/healthz
```

### 3. Python chunker service

```bash
cd chunker-service

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Levantar el servicio
python main.py
# Corre en http://localhost:8001
```

### 4. Go

```bash
cd rag-go

# Descargar dependencias
go mod tidy

# Verificar que compila
go build ./...
```

## Uso

### Indexar un documento

```bash
# Asegurarse que estén corriendo: Ollama, Qdrant, chunker-service

# Indexar con parámetros por defecto (chunk_size=512, overlap=80)
go run ./cmd/index -file ./data/pdfs/risk.pdf

# Con ID personalizado
go run ./cmd/index -file ./data/pdfs/risk.pdf -id risk-manual

# El progreso se muestra en tiempo real:
# 📄 Indexando: risk.pdf (id: risk.pdf)
# 🔪 Chunking... (chunk_size=512, overlap=80)
#    ✓ 47 chunks en 3.2s
# 🔢 Generando embeddings con nomic-embed-text...
#    [20/47] cache hits: 0 (0%)
#    [40/47] cache hits: 0 (0%)
#    [47/47] cache hits: 0 (0%)
#    ✓ Embeddings en 52.1s (cache hits: 0/47)
# 📤 Guardando en Qdrant...
#    ✓ Guardado en 0.3s
# ─────────────────────────────────────
# ✅ Indexado completado
#    Chunks    : 47
#    Cache hits: 0/47 (0%)
#    Tiempo    : 55.8s
```

### Segunda vez (con cache)

```bash
go run ./cmd/index -file ./data/pdfs/risk.pdf

# 🔢 Generando embeddings con nomic-embed-text...
#    [47/47] cache hits: 47 (100%)
#    ✓ Embeddings en 0.1s (cache hits: 47/47)
# ✅ Indexado en 3.4s  ← mucho más rápido
```

### Hacer preguntas

```bash
# Buscar en todos los documentos
go run ./cmd/query

# Buscar solo en un documento
go run ./cmd/query -doc risk.pdf

# Con más chunks de contexto
go run ./cmd/query -doc risk.pdf -k 7

# Ejemplo de sesión:
# ▶ ¿Cómo se gana una partida de Risk?
#
# 🤖 Respuesta:
# Para ganar una partida de Risk, un jugador debe conquistar todos los
# territorios del tablero [1]. El juego termina cuando un único jugador
# controla cada uno de los 42 territorios...
#
# 📚 Fuentes (búsqueda: 45ms):
#    [1] risk.pdf — pág. 4 (relevancia: 0.87)
#    [2] risk.pdf — pág. 3 (relevancia: 0.71)
```

## Configuración

Editar `internal/config/config.go` para cambiar:

```go
LLMModel:   "qwen2.5:7b-instruct"  // o "qwen2.5:3b" para más velocidad
EmbedModel: "nomic-embed-text"
DefaultTopK: 5                      // chunks a recuperar
```

## Modelos alternativos por velocidad vs calidad

| Modelo | Calidad | Velocidad CPU | RAM |
|--------|---------|---------------|-----|
| qwen2.5:7b-instruct | ⭐⭐⭐⭐⭐ | Medio | ~4.5GB |
| qwen2.5:3b | ⭐⭐⭐⭐ | Rápido | ~2.5GB |
| phi3:mini | ⭐⭐⭐ | Muy rápido | ~2.3GB |

## Roadmap

### Fase 1 (actual)
- ✅ Pipeline básico funcional
- ✅ Cache de embeddings
- ✅ Streaming de respuestas
- ✅ Filtrado por documento
- ✅ Citado de fuentes

### Fase 2 (próxima)
- [ ] Hierarchical chunking + parent-child retrieval
- [ ] Hybrid search (dense + BM25 sparse)
- [ ] Reranking con cross-encoder
- [ ] Multi-query retrieval