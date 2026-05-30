// Package embed maneja la generación de embeddings usando Ollama.
// Usa nomic-embed-text que es especializado para retrieval y mucho
// más rápido que pedirle embeddings al LLM de chat.
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"rag-go/internal/config"
)

// Embedder genera embeddings vectoriales de texto usando Ollama.
type Embedder struct {
	baseURL string
	model   config.EmbedModel
	client  *http.Client
}

// New crea un nuevo Embedder conectado a Ollama.
func New(baseURL string, model config.EmbedModel) *Embedder {
	return &Embedder{
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{},
	}
}

type embedRequest struct {
	Model  config.EmbedModel `json:"model"`
	Prompt string            `json:"prompt"`
}

type embedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Embed genera el embedding de un texto.
//
// El parámetro isQuery distingue si es un documento a indexar
// o una query del usuario. nomic-embed-text usa prefijos distintos
// para optimizar la búsqueda semántica:
//   - "search_document:" para chunks de documentos
//   - "search_query:" para queries del usuario
//
// Este prefixing mejora la calidad de retrieval porque el modelo
// fue entrenado explícitamente con esta distinción.
func (e *Embedder) Embed(ctx context.Context, text string, isQuery bool) ([]float32, error) {
	prefix := "search_document"
	if isQuery {
		prefix = "search_query"
	}

	reqBody := embedRequest{
		Model:  e.model,
		Prompt: prefix + ": " + text,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("serializar: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, e.baseURL+"/api/embeddings", bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("crear request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama no responde (¿está corriendo?): %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama respondió %d", resp.StatusCode)
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decodificar respuesta: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("ollama devolvió un embedding vacío (¿modelo %q instalado?)", e.model)
	}

	return result.Embedding, nil
}
