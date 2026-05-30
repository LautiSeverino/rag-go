// cmd/query es el comando interactivo para hacer preguntas al sistema RAG.
// Lee preguntas del usuario en un loop, busca en Qdrant y genera respuestas
// con streaming para que se vean los tokens en tiempo real.
//
// Uso:
//
//	go run ./cmd/query
//	go run ./cmd/query -doc risk.pdf -k 5
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"rag-go/internal/config"
	"rag-go/internal/embed"
	"rag-go/internal/llm"
	"rag-go/internal/store"
	"strings"
	"time"
)

func main() {
	docID := flag.String("doc", "", "Filtrar por documento (ej: risk.pdf). Vacío = busca en todos")
	topK := flag.Uint64("k", 5, "Número de chunks a recuperar")
	flag.Parse()

	cfg := config.Default()
	ctx := context.Background()

	// Inicializar componentes
	embedder := embed.New(cfg.OllamaURL, cfg.EmbedModel)
	llmClient := llm.New(cfg.OllamaURL, cfg.LLMModel)

	qdrantStore, err := store.New(cfg.QdrantHost, cfg.QdrantPort, cfg.CollectionName, cfg.VectorDims)
	if err != nil {
		log.Fatalf("❌ Qdrant: %v\n   ¿Está corriendo? docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant", err)
	}

	// Header
	fmt.Println("╔══════════════════════════════════╗")
	fmt.Println("║          RAG - Go + Qwen          ║")
	fmt.Println("╚══════════════════════════════════╝")
	if *docID != "" {
		fmt.Printf("📚 Buscando en: %s\n", *docID)
	} else {
		fmt.Println("📚 Buscando en todos los documentos")
	}
	fmt.Printf("🔍 Top-K: %d chunks\n", *topK)
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
		if err := handleQuery(ctx, question, *docID, *topK, embedder, llmClient, qdrantStore); err != nil {
			fmt.Printf("❌ Error: %v\n\n", err)
		}
	}
}

func handleQuery(
	ctx context.Context,
	question string,
	docID string,
	topK uint64,
	embedder *embed.Embedder,
	llmClient *llm.Client,
	qdrantStore *store.QdrantStore,
) error {

	totalStart := time.Now()

	// --- Paso 1: Embed de la query ---
	embedStart := time.Now()

	queryVec, err := embedder.Embed(ctx, question, true)
	if err != nil {
		return fmt.Errorf("embed query: %w", err)
	}

	embedTime := time.Since(embedStart)

	// --- Paso 2: Búsqueda en Qdrant ---
	searchStart := time.Now()

	results, err := qdrantStore.Search(ctx, queryVec, topK, docID)
	if err != nil {
		return fmt.Errorf("búsqueda: %w", err)
	}

	searchTime := time.Since(searchStart)

	if len(results) == 0 {
		fmt.Println("🔍 No encontré información relevante en los documentos indexados.")
		fmt.Println()
		return nil
	}

	// ===== DEBUG: CHUNKS RECUPERADOS =====
	fmt.Println("=== CHUNKS RECUPERADOS ===")

	for i, r := range results {
		fmt.Printf("\n[%d] score=%.3f page=%d\n",
			i+1,
			r.Score,
			r.Page,
		)

		text := r.Text
		if len(text) > 400 {
			text = text[:400]
		}

		fmt.Println(text)
	}

	fmt.Println("==========================")
	fmt.Println()

	// Filtrar resultados con score muy bajo
	const minScore = 0.4

	filtered := results[:0]
	for _, r := range results {
		if r.Score >= minScore {
			filtered = append(filtered, r)
		}
	}

	if len(filtered) == 0 {
		fmt.Printf("🔍 Encontré %d chunks pero con muy baja relevancia (score < %.1f).\n",
			len(results),
			minScore,
		)
		fmt.Println("   Probá reformular la pregunta.")
		fmt.Println()
		return nil
	}

	// --- Paso 3: Construir prompt ---
	searchChunks := make([]llm.SearchChunk, len(filtered))

	for i, r := range filtered {
		searchChunks[i] = llm.SearchChunk{
			Index:  i + 1,
			Text:   r.Text,
			Source: r.Source,
			Page:   r.Page,
			Score:  r.Score,
		}
	}

	prompt := llm.BuildPrompt(question, searchChunks)

	// ===== DEBUG: PROMPT =====
	fmt.Println("=== PROMPT ===")
	fmt.Println(prompt)
	fmt.Println("==============")
	fmt.Println()

	// --- Paso 4: Generar respuesta ---
	fmt.Println("🤖 Respuesta:")
	fmt.Println()

	llmStart := time.Now()

	tokenCh, errCh := llmClient.GenerateStream(ctx, prompt)

	for token := range tokenCh {
		fmt.Print(token)
	}

	if err := <-errCh; err != nil {
		return fmt.Errorf("generar respuesta: %w", err)
	}

	llmTime := time.Since(llmStart)

	fmt.Println()
	fmt.Println()

	// --- Métricas ---
	fmt.Printf("🔢 Embedding query: %v\n", embedTime)
	fmt.Printf("🔍 Búsqueda vectorial: %v\n", searchTime)
	fmt.Printf("🤖 Generación LLM: %v\n", llmTime)
	fmt.Printf("⏱ Total: %v\n", time.Since(totalStart))
	fmt.Println()

	// --- Fuentes ---
	fmt.Printf("📚 Fuentes (búsqueda: %dms):\n",
		searchTime.Milliseconds(),
	)

	for i, r := range filtered {
		fmt.Printf(
			"   [%d] %s — pág. %d (relevancia: %.2f)\n",
			i+1,
			r.Source,
			r.Page,
			r.Score,
		)
	}

	fmt.Println()

	return nil
}
