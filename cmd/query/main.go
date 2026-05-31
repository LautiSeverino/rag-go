// cmd/query/main.go — Con BM25 + RRF hybrid retrieval
// REEMPLAZAR main.go existente con este contenido

package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"rag-go/internal/bm25"
	"rag-go/internal/config"
	"rag-go/internal/embed"
	"rag-go/internal/llm"
	"rag-go/internal/store"
	"strings"
	"time"
)

func main() {
	docID := flag.String("doc", "", "Filtrar por documento (ej: teg.pdf)")
	topK := flag.Uint64("k", 20, "Chunks candidatos de Qdrant (se rerankeará a 5)")
	finalK := flag.Int("final-k", 5, "Chunks finales para el prompt tras reranking")
	debug := flag.Bool("debug", false, "Mostrar chunks recuperados y prompt")
	flag.Parse()

	cfg := config.Default()
	ctx := context.Background()

	// Inicializar componentes
	embedder := embed.New(cfg.OllamaURL, cfg.EmbedModel)
	llmClient := llm.New(cfg.OllamaURL, cfg.LLMModel)

	qdrantStore, err := store.New(cfg.QdrantHost, cfg.QdrantPort, cfg.CollectionName, cfg.VectorDims)
	if err != nil {
		log.Fatalf("❌ Qdrant: %v", err)
	}

	// Cargar índice BM25 si existe
	var bm25Idx *bm25.Index
	if *docID != "" {
		bm25Path := filepath.Join(cfg.DataDir, "bm25", *docID+".json")
		if idx, err := bm25.Load(bm25Path); err == nil {
			bm25Idx = idx
			fmt.Printf("✓ Índice BM25 cargado (%d docs)\n", len(idx.Docs))
		} else {
			fmt.Printf("⚠️  Sin índice BM25 (re-indexar para activarlo)\n")
		}
	}

	// Header
	fmt.Println("╔══════════════════════════════════╗")
	fmt.Println("║          RAG - Go + Qwen          ║")
	fmt.Println("╚══════════════════════════════════╝")

	retrieval := "Dense (Qdrant)"
	if bm25Idx != nil {
		retrieval = "Hybrid (Dense + BM25 → RRF)"
	}
	fmt.Printf("🔍 Retrieval: %s\n", retrieval)
	if *docID != "" {
		fmt.Printf("📚 Documento: %s\n", *docID)
	}
	fmt.Printf("🔢 Candidatos: %d → Prompt: %d\n", *topK, *finalK)
	fmt.Println()
	fmt.Println("Escribí tu pregunta (Ctrl+C para salir):")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("▶ ")
		if !scanner.Scan() {
			break
		}

		question := strings.TrimSpace(scanner.Text())
		if question == "" {
			continue
		}

		fmt.Println()
		if err := handleQuery(
			ctx, question, *docID, *topK, *finalK, *debug,
			embedder, llmClient, qdrantStore, bm25Idx,
		); err != nil {
			fmt.Printf("❌ Error: %v\n\n", err)
		}
	}
}

func handleQuery(
	ctx context.Context,
	question string,
	docID string,
	topK uint64,
	finalK int,
	debug bool,
	embedder *embed.Embedder,
	llmClient *llm.Client,
	qdrantStore *store.QdrantStore,
	bm25Idx *bm25.Index,
) error {
	queryStart := time.Now()

	// 1. Embed query
	queryVec, err := embedder.Embed(ctx, question, true)
	if err != nil {
		return fmt.Errorf("embed query: %w", err)
	}
	embedTime := time.Since(queryStart)

	// 2. Dense search en Qdrant (traer más candidatos para reranking)
	searchStart := time.Now()
	qdrantResults, err := qdrantStore.Search(ctx, queryVec, topK, docID)
	if err != nil {
		return fmt.Errorf("búsqueda Qdrant: %w", err)
	}
	searchTime := time.Since(searchStart)

	if len(qdrantResults) == 0 {
		fmt.Println("🔍 No encontré chunks en Qdrant. ¿Indexaste el documento?")
		fmt.Println()
		return nil
	}

	// 3. Convertir resultados de Qdrant al formato de BM25
	denseForFusion := make([]struct {
		ID         string
		DenseScore float32
		Text       string
		Source     string
		Page       int
	}, len(qdrantResults))

	for i, r := range qdrantResults {
		denseForFusion[i].ID = r.ID
		denseForFusion[i].DenseScore = r.Score
		denseForFusion[i].Text = r.Text
		denseForFusion[i].Source = r.Source
		denseForFusion[i].Page = r.Page
	}

	// 4. Hybrid retrieval con BM25 + RRF
	rerankStart := time.Now()
	fusedResults := bm25.FuseResults(denseForFusion, bm25Idx, question, finalK)
	rerankTime := time.Since(rerankStart)

	if debug {
		fmt.Println("=== CHUNKS RERANKEADOS ===")
		for i, r := range fusedResults {
			fmt.Printf("[%d] dense_rank=%d bm25_rank=%d rrf=%.4f dense=%.3f bm25=%.2f\n",
				i+1, r.DenseRank+1, r.BM25Rank+1, r.RRFScore, r.DenseScore, r.BM25Score)
			fmt.Printf("    %s\n", r.Text[:min(80, len(r.Text))])
		}
		fmt.Println("==========================")
		fmt.Println()
	}

	if len(fusedResults) == 0 {
		fmt.Println("🔍 Sin resultados relevantes.")
		fmt.Println()
		return nil
	}

	// 5. Construir prompt
	searchChunks := make([]llm.SearchChunk, len(fusedResults))
	for i, r := range fusedResults {
		// Buscar source/page del resultado fusionado en resultados originales
		source := ""
		page := 0
		for _, orig := range qdrantResults {
			if orig.ID == r.ID {
				source = orig.Source
				page = orig.Page
				break
			}
		}
		if source == "" {
			source = r.Source
			page = r.Page
		}
		searchChunks[i] = llm.SearchChunk{
			Index:  i + 1,
			Text:   r.Text,
			Source: source,
			Page:   page,
			Score:  r.DenseScore,
		}
	}

	if debug {
		prompt := llm.BuildPrompt(question, searchChunks)
		fmt.Println("=== PROMPT ===")
		fmt.Println(prompt)
		fmt.Println("==============")
		fmt.Println()
	}

	// 6. Generar respuesta con streaming
	prompt := llm.BuildPrompt(question, searchChunks)
	fmt.Println("🤖 Respuesta:")
	fmt.Println()

	genStart := time.Now()
	tokenCh, errCh := llmClient.GenerateStream(ctx, prompt)
	for token := range tokenCh {
		fmt.Print(token)
	}
	if err := <-errCh; err != nil {
		return fmt.Errorf("generación LLM: %w", err)
	}
	genTime := time.Since(genStart)

	fmt.Println()
	fmt.Println()

	// 7. Mostrar métricas y fuentes
	fmt.Printf("🔢 Embed: %dms | 🔍 Qdrant: %dms | ⚖️  Rerank: %dms | 🤖 LLM: %s\n",
		embedTime.Milliseconds(), searchTime.Milliseconds(),
		rerankTime.Microseconds(), genTime.Round(time.Second))
	fmt.Println()

	fmt.Println("📚 Fuentes:")
	for _, r := range fusedResults {
		source := r.Source
		page := r.Page
		for _, orig := range qdrantResults {
			if orig.ID == r.ID {
				source = orig.Source
				page = orig.Page
				break
			}
		}
		fmt.Printf("   pág.%d — dense_rank:%d bm25_rank:%d — %s\n",
			page, r.DenseRank+1, r.BM25Rank+1, source)
	}
	fmt.Println()

	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
