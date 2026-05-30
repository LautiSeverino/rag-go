// Package chunker implementa el cliente HTTP para comunicarse con
// el Python chunker service. Go no hace el chunking directamente;
// delega esa responsabilidad al servicio Python que tiene mejor
// ecosistema para procesamiento de texto y PDFs.
package chunker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// ChunkRequest es lo que Go le manda al servicio Python.
type ChunkRequest struct {
	FilePath  string `json:"file_path"`
	Strategy  string `json:"strategy"`   // "recursive" en Fase 1
	ChunkSize int    `json:"chunk_size"` // tokens
	Overlap   int    `json:"overlap"`    // tokens
	DocID     string `json:"doc_id"`
}

// ChunkMetadata son los metadatos que acompañan a cada chunk.
// Se guardan en Qdrant junto al vector y permiten filtrado y citado de fuentes.
type ChunkMetadata struct {
	DocID       string `json:"doc_id"`
	Source      string `json:"source"`
	Page        int    `json:"page"`
	ChunkIndex  int    `json:"chunk_index"`
	TotalChunks int    `json:"total_chunks"`
}

// Chunk representa un fragmento de texto listo para embeddear e indexar.
type Chunk struct {
	Text     string        `json:"text"`
	Metadata ChunkMetadata `json:"metadata"`
}

// Client se comunica con el Python chunker service vía HTTP.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient crea un cliente para el chunker service.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			// Los PDFs grandes pueden tardar en procesar.
			// 5 minutos es suficiente para cualquier libro.
			Timeout: 5 * time.Minute,
		},
	}
}

// Chunk envía un PDF al servicio Python y recibe los chunks procesados.
func (c *Client) Chunk(ctx context.Context, req ChunkRequest) ([]Chunk, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("serializar request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx, http.MethodPost, c.baseURL+"/chunk", bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("crear request HTTP: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llamar chunker service (¿está corriendo en %s?): %w", c.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errBody struct {
			Detail string `json:"detail"`
		}
		json.NewDecoder(resp.Body).Decode(&errBody)
		return nil, fmt.Errorf("chunker service error %d: %s", resp.StatusCode, errBody.Detail)
	}

	var result struct {
		Chunks []Chunk `json:"chunks"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decodificar respuesta: %w", err)
	}

	return result.Chunks, nil
}

// HealthCheck verifica que el servicio Python esté corriendo.
func (c *Client) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("chunker service no responde en %s: %w", c.baseURL, err)
	}
	resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("chunker service respondió con status %d", resp.StatusCode)
	}
	return nil
}
