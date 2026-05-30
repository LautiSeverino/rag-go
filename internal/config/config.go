package config

type EmbedModel string

const (
	EmbedModelNomicText EmbedModel = "nomic-embed-text"
)

type LLMModel string

const (
	LLMModelQwen25 LLMModel = "qwen2.5:7b-instruct"
)

// Config centraliza toda la configuración del sistema RAG.
// Los valores por defecto apuntan a servicios corriendo localmente.
type Config struct {
	// Ollama
	OllamaURL string
	EmbedModel
	LLMModel

	// Qdrant
	QdrantHost     string
	QdrantPort     int
	CollectionName string
	VectorDims     uint64

	// Python chunker service
	ChunkerURL string

	// Storage
	CacheDir string
	DataDir  string

	// Retrieval
	DefaultTopK int
}

func Default() *Config {
	return &Config{
		OllamaURL:      "http://localhost:11434",
		EmbedModel:     "nomic-embed-text",
		LLMModel:       "qwen2.5:7b-instruct",
		QdrantHost:     "localhost",
		QdrantPort:     6334,
		CollectionName: "rag_docs",
		VectorDims:     768, // nomic-embed-text genera vectores de 768 dims
		ChunkerURL:     "http://localhost:8001",
		CacheDir:       "./data/cache",
		DataDir:        "./data",
		DefaultTopK:    5,
	}
}
