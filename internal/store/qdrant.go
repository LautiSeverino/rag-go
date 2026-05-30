// Package store maneja la comunicación con Qdrant para almacenar
// y buscar vectores. Usa gRPC (puerto 6334) en lugar de HTTP REST
// porque es más eficiente para llamadas frecuentes.
package store

import (
	"context"
	"fmt"

	"github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Point representa un vector con su payload para insertar en Qdrant.
type Point struct {
	ID      string
	Vector  []float32
	Payload map[string]any
}

// SearchResult es lo que devuelve Qdrant después de una búsqueda.
type SearchResult struct {
	ID         string
	Score      float32
	Text       string
	Source     string
	Page       int
	DocID      string
	ChunkIndex int
}

// QdrantStore es el wrapper sobre el cliente gRPC de Qdrant.
type QdrantStore struct {
	collectionsClient qdrant.CollectionsClient
	pointsClient      qdrant.PointsClient
	client            *qdrant.Client
	collectionName    string
	dimensions        uint64
}

// New crea una conexión a Qdrant y devuelve el store.
func New(host string, port int, collectionName string, dims uint64) (*QdrantStore, error) {
	target := fmt.Sprintf("%s:%d", host, port)

	conn, err := grpc.NewClient(
		target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("conectar gRPC a Qdrant en %s: %w", target, err)
	}

	client, err := qdrant.NewClient(&qdrant.Config{
		Host: host,
		Port: port,
	})
	if err != nil {
		return nil, fmt.Errorf("crear cliente Qdrant en %s: %w", target, err)
	}

	return &QdrantStore{
		collectionsClient: qdrant.NewCollectionsClient(conn),
		pointsClient:      qdrant.NewPointsClient(conn),
		client:            client,
		collectionName:    collectionName,
		dimensions:        dims,
	}, nil
}

// EnsureCollection crea la colección si no existe.
// Si ya existe, no hace nada. Es seguro llamarlo siempre al inicio.
//
// Configuración:
//   - Cosine distance: mejor para embeddings de texto
//   - HNSW m=16: balance entre RAM y precisión
//   - INT8 quantization: 4x menos RAM con ~1-2% pérdida de calidad
func (s *QdrantStore) EnsureCollection(ctx context.Context) error {
	collections, err := s.collectionsClient.List(ctx, &qdrant.ListCollectionsRequest{})
	if err != nil {
		return fmt.Errorf("listar colecciones: %w", err)
	}

	for _, col := range collections.Collections {
		if col.Name == s.collectionName {
			return nil // ya existe
		}
	}

	distance := qdrant.Distance_Cosine
	trueVal := true
	m := uint64(16)
	efConstruct := uint64(200)

	_, err = s.collectionsClient.Create(ctx, &qdrant.CreateCollection{
		CollectionName: s.collectionName,
		VectorsConfig: &qdrant.VectorsConfig{
			Config: &qdrant.VectorsConfig_Params{
				Params: &qdrant.VectorParams{
					Size:     s.dimensions,
					Distance: distance,
				},
			},
		},
		HnswConfig: &qdrant.HnswConfigDiff{
			M:           &m,
			EfConstruct: &efConstruct,
		},
		QuantizationConfig: &qdrant.QuantizationConfig{
			Quantization: &qdrant.QuantizationConfig_Scalar{
				Scalar: &qdrant.ScalarQuantization{
					Type:      qdrant.QuantizationType_Int8,
					AlwaysRam: &trueVal,
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("crear colección: %w", err)
	}

	fmt.Printf("✓ Colección '%s' creada en Qdrant\n", s.collectionName)
	return nil
}

// Upsert inserta o actualiza puntos en Qdrant.
// Se llama en batches desde el pipeline de indexación.
func (s *QdrantStore) Upsert(ctx context.Context, points []Point) error {
	qPoints := make([]*qdrant.PointStruct, len(points))

	for i, p := range points {
		payload := make(map[string]*qdrant.Value)

		for k, v := range p.Payload {
			switch val := v.(type) {
			case string:
				payload[k] = &qdrant.Value{
					Kind: &qdrant.Value_StringValue{StringValue: val},
				}
			case int:
				payload[k] = &qdrant.Value{
					Kind: &qdrant.Value_IntegerValue{IntegerValue: int64(val)},
				}
			case float64:
				payload[k] = &qdrant.Value{
					Kind: &qdrant.Value_DoubleValue{DoubleValue: val},
				}
			}
		}

		qPoints[i] = &qdrant.PointStruct{
			Id: &qdrant.PointId{
				PointIdOptions: &qdrant.PointId_Uuid{Uuid: p.ID},
			},
			Vectors: &qdrant.Vectors{
				VectorsOptions: &qdrant.Vectors_Vector{
					Vector: &qdrant.Vector{Data: p.Vector},
				},
			},
			Payload: payload,
		}
	}

	wait := true
	_, err := s.pointsClient.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collectionName,
		Points:         qPoints,
		Wait:           &wait,
	})
	if err != nil {
		return fmt.Errorf("upsert: %w", err)
	}

	return nil
}

// Search busca los K vectores más similares al query vector.
// Si docID no es vacío, filtra para buscar solo dentro de ese documento.
func (s *QdrantStore) Search(ctx context.Context, queryVector []float32, topK uint64, docID string) ([]SearchResult, error) {
	req := &qdrant.SearchPoints{
		CollectionName: s.collectionName,
		Vector:         queryVector,
		Limit:          topK,
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true},
		},
	}

	// Filtrar por documento si se especifica
	if docID != "" {
		req.Filter = &qdrant.Filter{
			Must: []*qdrant.Condition{
				{
					ConditionOneOf: &qdrant.Condition_Field{
						Field: &qdrant.FieldCondition{
							Key: "doc_id",
							Match: &qdrant.Match{
								MatchValue: &qdrant.Match_Keyword{Keyword: docID},
							},
						},
					},
				},
			},
		}
	}

	resp, err := s.pointsClient.Search(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("buscar en Qdrant: %w", err)
	}

	results := make([]SearchResult, 0, len(resp.Result))

	for _, r := range resp.Result {
		sr := SearchResult{Score: r.Score}

		if id, ok := r.Id.PointIdOptions.(*qdrant.PointId_Uuid); ok {
			sr.ID = id.Uuid
		}

		if v, ok := r.Payload["text"]; ok {
			sr.Text = v.GetStringValue()
		}
		if v, ok := r.Payload["source"]; ok {
			sr.Source = v.GetStringValue()
		}
		if v, ok := r.Payload["doc_id"]; ok {
			sr.DocID = v.GetStringValue()
		}
		if v, ok := r.Payload["page"]; ok {
			sr.Page = int(v.GetIntegerValue())
		}
		if v, ok := r.Payload["chunk_index"]; ok {
			sr.ChunkIndex = int(v.GetIntegerValue())
		}

		results = append(results, sr)
	}

	return results, nil
}

// CreatePayloadIndex crea un índice en un campo del payload para
// acelerar los filtros. Llamar después de indexar un documento.
func (s *QdrantStore) CreatePayloadIndex(ctx context.Context, fieldName string) error {
	_, err := s.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
		CollectionName: s.collectionName,
		FieldName:      fieldName,
		FieldType:      qdrant.FieldType_FieldTypeKeyword.Enum(),
	})

	if err != nil {
		return fmt.Errorf("crear índice payload: %w", err)
	}

	return nil
}
