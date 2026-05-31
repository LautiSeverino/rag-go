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

const temperature = 0.1 // Baja temperatura para respuestas más precisas en RAG
const numCtx = 4096     // Ventana de contexto amplia para manejar prompts largos con mucho contexto

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
				"temperature": temperature, // Baja temperatura para respuestas más precisas en RAG
				"num_ctx":     numCtx,      // Ventana de contexto
			},
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			errCh <- fmt.Errorf("serializar request: %w", err)
			return
		}

		req, err := http.NewRequestWithContext(
			ctx, http.MethodPost, c.baseURL+"/api/generate", bytes.NewReader(body),
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

// BuildPrompt construye el prompt RAG con el contexto recuperado.
//
// El diseño del prompt es importante:
//   - Instruye al modelo a responder SOLO con el contexto dado
//   - Pide que cite el número de fragmento cuando usa info de él
//   - Pide que diga "no tengo esa información" si no está en el contexto
//     (evita alucinaciones)
func BuildPrompt(question string, chunks []SearchChunk) string {
	var sb strings.Builder

	sb.WriteString(`Eres un asistente experto en responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS ESTRICTAS:
1. Responde SOLO usando información del CONTEXTO proporcionado
2. NUNCA inventes, supongas o agregues información externa
3. Responde en el MISMO IDIOMA de la PREGUNTA
   - Pregunta en español → respuesta en español
   - Pregunta en inglés → respuesta en inglés
4. Si el contexto tiene información dispersa en varios fragmentos, combínala de forma coherente
5. Si la pregunta pide "cómo" o un proceso paso a paso, incluye TODOS los pasos mencionados en el contexto
6. Si NO HAY información suficiente, responde: "No tengo suficiente información en el contexto para responder esta pregunta."
7. Sé preciso, completo y detallado. No omitas pasos ni información relevante.
8. Usa SOLO datos del contexto
9. Cuando uses información de un fragmento, cita su número entre corchetes, por ejemplo: [1], [2], [3]

`)

	sb.WriteString("--- CONTEXTO ---\n\n")

	for _, chunk := range chunks {
		sb.WriteString(fmt.Sprintf(
			"[%d] (Fuente: %s, página %d)\n",
			chunk.Index,
			chunk.Source,
			chunk.Page,
		))
		sb.WriteString(chunk.Text)
		sb.WriteString("\n\n")
	}

	sb.WriteString("--- FIN DEL CONTEXTO ---\n\n")
	sb.WriteString(fmt.Sprintf("PREGUNTA: %s\n\n", question))
	sb.WriteString("RESPUESTA:")

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
