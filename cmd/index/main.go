// cmd/index es el comando para indexar un PDF en el sistema RAG.
//
// Uso:
//
//	go run ./cmd/index -file riesgo.pdf
//	go run ./cmd/index -file poole.pdf -id poole-algebra -chunk-size 512 -overlap 80
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"rag-go/data/cache"
	"rag-go/internal/chunker"
	"rag-go/internal/config"
	"rag-go/internal/embed"
	"rag-go/internal/store"
	"time"

	"github.com/google/uuid"
)

func main() {
	filePath := flag.String("file", "", "Ruta al PDF a indexar (requerido)")
	docID := flag.String("id", "", "ID del documento (default: nombre del archivo)")
	chunkSize := flag.Int("chunk-size", 512, "Tamaño de chunk en tokens")
	overlap := flag.Int("overlap", 80, "Overlap entre chunks en tokens")
	flag.Parse()

	if *filePath == "" {
		fmt.Fprintln(os.Stderr, "Error: -file es requerido")
		fmt.Fprintln(os.Stderr, "Uso: go run ./cmd/index -file documento.pdf")
		os.Exit(1)
	}

	if *docID == "" {
		*docID = filepath.Base(*filePath)
	}

	cfg := config.Default()
	ctx := context.Background()

	start := time.Now()
	fmt.Printf("📄 Indexando: %s (id: %s)\n", *filePath, *docID)
	fmt.Println()

	// --- Inicializar componentes ---

	chunkerClient := chunker.NewClient(cfg.ChunkerURL)

	// Verificar que el servicio Python esté corriendo
	if err := chunkerClient.HealthCheck(ctx); err != nil {
		log.Fatalf("❌ %v\n   Corré: cd chunker-service && python main.py", err)
	}

	embedder := embed.New(cfg.OllamaURL, cfg.EmbedModel)

	embCache, err := cache.New(filepath.Join(cfg.CacheDir, "embeddings.db"))
	if err != nil {
		log.Fatalf("❌ Cache: %v", err)
	}
	defer embCache.Close()

	qdrantStore, err := store.New(cfg.QdrantHost, cfg.QdrantPort, cfg.CollectionName, cfg.VectorDims)
	if err != nil {
		log.Fatalf("❌ Qdrant: %v\n   ¿Está corriendo? docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant", err)
	}

	if err := qdrantStore.EnsureCollection(ctx); err != nil {
		log.Fatalf("❌ Crear colección: %v", err)
	}

	// --- Paso 1: Chunking via Python service ---

	fmt.Printf("🔪 Chunking... (chunk_size=%d, overlap=%d)\n", *chunkSize, *overlap)
	chunkStart := time.Now()

	absPath, err := filepath.Abs(*filePath)
	if err != nil {
		log.Fatalf("❌ Resolver ruta absoluta: %v", err)
	}
	chunks, err := chunkerClient.Chunk(ctx, chunker.ChunkRequest{
		FilePath:  absPath,
		Strategy:  "section_aware", // estrategia que intenta respetar secciones y párrafos, además de usar overlap para mejorar contexto
		ChunkSize: *chunkSize,
		Overlap:   *overlap,
		DocID:     *docID,
	})
	if err != nil {
		log.Fatalf("❌ Chunking: %v", err)
	}

	fmt.Printf("   ✓ %d chunks en %.1fs\n\n", len(chunks), time.Since(chunkStart).Seconds())

	for _, chunk := range chunks {
		fmt.Printf(
			"chunk=%d page=%d len=%d\n",
			chunk.Metadata.ChunkIndex,
			chunk.Metadata.Page,
			len(chunk.Text),
		)
	}

	// --- Paso 2: Embeddings con cache ---

	fmt.Printf("🔢 Generando embeddings con %s...\n", cfg.EmbedModel)
	embedStart := time.Now()

	cacheHits := 0
	points := make([]store.Point, 0, len(chunks))

	for i, chunk := range chunks {
		hash := cache.HashText(chunk.Text)

		var embedding []float32

		if emb, ok := embCache.Get(hash); ok {
			embedding = emb
			cacheHits++
		} else {
			emb, err := embedder.Embed(ctx, chunk.Text, false)
			if err != nil {
				log.Fatalf("❌ Embed chunk %d: %v", i, err)
			}
			if err := embCache.Set(hash, emb); err != nil {
				log.Printf("⚠️  No se pudo cachear chunk %d: %v", i, err)
			}
			embedding = emb
		}

		points = append(points, store.Point{
			ID:     uuid.New().String(),
			Vector: embedding,
			Payload: map[string]any{
				"text":        chunk.Text,
				"doc_id":      chunk.Metadata.DocID,
				"source":      chunk.Metadata.Source,
				"page":        chunk.Metadata.Page,
				"chunk_index": chunk.Metadata.ChunkIndex,
			},
		})

		// Mostrar progreso cada 20 chunks
		if (i+1)%20 == 0 || i+1 == len(chunks) {
			cacheRate := float64(cacheHits) / float64(i+1) * 100
			fmt.Printf("   [%d/%d] cache hits: %d (%.0f%%)\n",
				i+1, len(chunks), cacheHits, cacheRate)
		}
	}

	fmt.Printf("   ✓ Embeddings en %.1fs (cache hits: %d/%d)\n\n",
		time.Since(embedStart).Seconds(), cacheHits, len(chunks))

	// --- Paso 3: Upsert en Qdrant por batches ---

	fmt.Println("📤 Guardando en Qdrant...")
	qdrantStart := time.Now()

	const batchSize = 50
	for i := 0; i < len(points); i += batchSize {
		end := min(i+batchSize, len(points))
		if err := qdrantStore.Upsert(ctx, points[i:end]); err != nil {
			log.Fatalf("❌ Upsert batch %d-%d: %v", i, end, err)
		}
	}

	// Crear índice en doc_id para que los filtros sean rápidos
	_ = qdrantStore.CreatePayloadIndex(ctx, "doc_id")

	fmt.Printf("   ✓ Guardado en %.1fs\n\n", time.Since(qdrantStart).Seconds())

	// --- Resumen ---

	total := time.Since(start)
	fmt.Println("─────────────────────────────────────")
	fmt.Printf("✅ Indexado completado\n")
	fmt.Printf("   Documento : %s\n", *docID)
	fmt.Printf("   Chunks    : %d\n", len(chunks))
	fmt.Printf("   Cache hits: %d/%d (%.0f%%)\n",
		cacheHits, len(chunks), float64(cacheHits)/float64(len(chunks))*100)
	fmt.Printf("   Tiempo    : %.1fs\n", total.Seconds())
	fmt.Println("─────────────────────────────────────")
	fmt.Printf("\nListo para consultar:\n")
	fmt.Printf("  go run ./cmd/query -doc %s\n", *docID)
}
