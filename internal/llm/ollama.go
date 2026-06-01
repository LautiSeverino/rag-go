// Package llm maneja la generación de respuestas con streaming usando Ollama.
// El streaming es crítico para la experiencia de usuario: en lugar de esperar
// 20-30 segundos sin feedback, el usuario ve los tokens aparecer desde el primer segundo.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"rag-go/internal/config"
	"strings"
)

// Client se comunica con Ollama para generación de texto.
type Client struct {
	baseURL string
	model   config.LLMModel
	client  *http.Client
}

// New crea un cliente LLM para Ollama.
func New(baseURL string, model config.LLMModel) *Client {
	return &Client{
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{},
	}
}

type generateRequest struct {
	Model   config.LLMModel `json:"model"`
	Prompt  string          `json:"prompt"`
	Stream  bool            `json:"stream"`
	Options map[string]any  `json:"options,omitempty"`
}

type generateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// GenerateStream envía el prompt a Ollama y devuelve dos canales:
//   - tokenCh: recibe los tokens a medida que se generan (para imprimir en tiempo real)
//   - errCh: recibe un error si algo falla, nil si todo salió bien
//
// El caller debe consumir tokenCh hasta que se cierre, y luego leer errCh.
func (c *Client) GenerateStream(ctx context.Context, prompt string) (<-chan string, <-chan error) {
	tokenCh := make(chan string)
	errCh := make(chan error, 1)

	go func() {
		defer close(tokenCh)
		defer close(errCh)

		reqBody := generateRequest{
			Model:  c.model,
			Prompt: prompt,
			Stream: true,
			Options: map[string]any{
				"temperature":    0.0, // 0 = completamente determinista, sin creatividad
				"num_ctx":        4096,
				"num_predict":    512, // limitar longitud: respuestas cortas y precisas
				"top_p":          0.9,
				"repeat_penalty": 1.1,
			},
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			errCh <- fmt.Errorf("serializar request: %w", err)
			return
		}

		req, err := http.NewRequestWithContext(
			ctx, "POST", c.baseURL+"/api/generate", bytes.NewReader(body),
		)
		if err != nil {
			errCh <- fmt.Errorf("crear request: %w", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.client.Do(req)
		if err != nil {
			errCh <- fmt.Errorf("ollama no responde: %w", err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			errCh <- fmt.Errorf("ollama respondió %d", resp.StatusCode)
			return
		}

		// Ollama con stream=true devuelve una línea JSON por token
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			var gen generateResponse
			if err := json.Unmarshal(line, &gen); err != nil {
				continue
			}

			if gen.Response != "" {
				select {
				case <-ctx.Done():
					return
				case tokenCh <- gen.Response:
				}
			}

			if gen.Done {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			errCh <- fmt.Errorf("leer stream: %w", err)
		}
	}()

	return tokenCh, errCh
}

// BuildPrompt construye el prompt RAG con instrucciones estrictas de grounding.
//
// Diseño del prompt:
// - Instrucción clara de SOLO usar el contexto dado
// - Prohibición explícita de agregar información propia
// - Para listas/enumeraciones: copiar exactamente del documento
// - Frase de fallback exacta cuando la info no está
func BuildPrompt(question string, chunks []SearchChunk) string {
	var sb strings.Builder

	sb.WriteString("Sos un asistente que responde preguntas basándose EXCLUSIVAMENTE en el contexto dado.\n\n")
	sb.WriteString("REGLAS ESTRICTAS:\n")
	sb.WriteString("1. Usá SOLO la información que aparece textualmente en el contexto.\n")
	sb.WriteString("2. NO agregues información propia, suposiciones ni elaboraciones.\n")
	sb.WriteString("3. Si la pregunta pide una lista o enumeración, copiá exactamente los elementos del contexto.\n")
	sb.WriteString("4. Si la información no está en el contexto, respondé exactamente: \"No encontré esa información en los documentos.\"\n")
	sb.WriteString("5. Citá el número de fragmento al usar información de él, ej: [1], [2].\n")
	sb.WriteString("6. Respondé en el mismo idioma de la pregunta.\n\n")

	sb.WriteString("--- CONTEXTO ---\n\n")
	for _, chunk := range chunks {
		sb.WriteString(fmt.Sprintf("[%d] (Fuente: %s, página %d)\n", chunk.Index, chunk.Source, chunk.Page))
		sb.WriteString(chunk.Text)
		sb.WriteString("\n\n")
	}

	sb.WriteString("--- FIN DEL CONTEXTO ---\n\n")
	sb.WriteString(fmt.Sprintf("Pregunta: %s\n\n", question))
	sb.WriteString("Respuesta (basada únicamente en el contexto):")

	return sb.String()
}

// SearchChunk es lo que llega del retrieval para construir el prompt.
type SearchChunk struct {
	Index  int
	Text   string
	Source string
	Page   int
	Score  float32
}
