// cmd/index/main.go — Con BM25 index generation
// REEMPLAZAR main.go existente con este contenido

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"rag-go/data/cache"
	"rag-go/internal/bm25"
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
	chunkSize := flag.Int("chunk-size", 512, "Tamaño de chunk")
	overlap := flag.Int("overlap", 80, "Overlap entre chunks")
	strategy := flag.String("strategy", "section_aware", "Estrategia de chunking: section_aware | recursive")
	flag.Parse()

	if *filePath == "" {
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

	// Inicializar componentes
	chunkerClient := chunker.NewClient(cfg.ChunkerURL)
	if err := chunkerClient.HealthCheck(ctx); err != nil {
		log.Fatalf("❌ Chunker service: %v\n   Corré: cd chunker-service && python main.py", err)
	}

	embedder := embed.New(cfg.OllamaURL, cfg.EmbedModel)

	embCache, err := cache.New(filepath.Join(cfg.CacheDir, "embeddings.db"))
	if err != nil {
		log.Fatalf("❌ Cache: %v", err)
	}
	defer embCache.Close()

	qdrantStore, err := store.New(cfg.QdrantHost, cfg.QdrantPort, cfg.CollectionName, cfg.VectorDims)
	if err != nil {
		log.Fatalf("❌ Qdrant: %v", err)
	}
	if err := qdrantStore.EnsureCollection(ctx); err != nil {
		log.Fatalf("❌ Colección: %v", err)
	}

	// --- Paso 1: Chunking ---
	fmt.Printf("🔪 Chunking... (strategy=%s, chunk_size=%d, overlap=%d)\n", *strategy, *chunkSize, *overlap)
	chunkStart := time.Now()

	chunks, err := chunkerClient.Chunk(ctx, chunker.ChunkRequest{
		FilePath:  *filePath,
		Strategy:  *strategy,
		ChunkSize: *chunkSize,
		Overlap:   *overlap,
		DocID:     *docID,
	})
	if err != nil {
		log.Fatalf("❌ Chunking: %v", err)
	}
	fmt.Printf("   ✓ %d chunks en %.1fs\n\n", len(chunks), time.Since(chunkStart).Seconds())

	// --- Paso 2: Embeddings + BM25 prep ---
	fmt.Printf("🔢 Embeddings con %s...\n", cfg.EmbedModel)
	embedStart := time.Now()

	cacheHits := 0
	points := make([]store.Point, 0, len(chunks))
	bm25Docs := make([]struct{ ID, Text string }, 0, len(chunks))

	for i, chunk := range chunks {
		chunkUUID := uuid.New().String()
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
			embCache.Set(hash, emb)
			embedding = emb
		}

		points = append(points, store.Point{
			ID:     chunkUUID,
			Vector: embedding,
			Payload: map[string]any{
				"text":        chunk.Text,
				"doc_id":      chunk.Metadata.DocID,
				"source":      chunk.Metadata.Source,
				"page":        chunk.Metadata.Page,
				"chunk_index": chunk.Metadata.ChunkIndex,
			},
		})

		// Guardar para BM25 (misma UUID que en Qdrant)
		bm25Docs = append(bm25Docs, struct{ ID, Text string }{
			ID:   chunkUUID,
			Text: chunk.Text,
		})

		if (i+1)%10 == 0 || i+1 == len(chunks) {
			fmt.Printf("   [%d/%d] cache hits: %d (%.0f%%)\n",
				i+1, len(chunks), cacheHits,
				float64(cacheHits)/float64(i+1)*100)
		}
	}

	fmt.Printf("   ✓ %.1fs (cache: %d/%d)\n\n", time.Since(embedStart).Seconds(), cacheHits, len(chunks))

	// --- Paso 3: Upsert en Qdrant ---
	fmt.Println("📤 Guardando en Qdrant...")
	const batchSize = 50
	for i := 0; i < len(points); i += batchSize {
		end := i + batchSize
		if end > len(points) {
			end = len(points)
		}
		if err := qdrantStore.Upsert(ctx, points[i:end]); err != nil {
			log.Fatalf("❌ Upsert: %v", err)
		}
	}
	qdrantStore.CreatePayloadIndex(ctx, "doc_id")
	fmt.Println("   ✓ Guardado")

	// --- Paso 4: Construir y guardar índice BM25 ---
	fmt.Println("📑 Construyendo índice BM25...")
	bm25Idx := bm25.Build(bm25Docs, *docID)

	bm25Dir := filepath.Join(cfg.DataDir, "bm25")
	os.MkdirAll(bm25Dir, 0755)
	bm25Path := filepath.Join(bm25Dir, *docID+".json")

	if err := bm25Idx.Save(bm25Path); err != nil {
		log.Printf("⚠️  No se pudo guardar índice BM25: %v", err)
	} else {
		fmt.Printf("   ✓ Índice BM25 guardado: %s\n", bm25Path)
	}

	// --- Resumen ---
	fmt.Println()
	fmt.Println("─────────────────────────────────────")
	fmt.Printf("✅ Indexado completado\n")
	fmt.Printf("   Documento : %s\n", *docID)
	fmt.Printf("   Chunks    : %d\n", len(chunks))
	fmt.Printf("   Cache hits: %d/%d\n", cacheHits, len(chunks))
	fmt.Printf("   Tiempo    : %.1fs\n", time.Since(start).Seconds())
	fmt.Println("─────────────────────────────────────")
	fmt.Printf("\nConsultar:\n  go run ./cmd/query -doc %s\n", *docID)
}
