// Package bm25 implementa BM25 ranking para reranking local de chunks.
//
// BM25 resuelve el problema central del sistema: la búsqueda densa
// (nomic-embed-text) no puede distinguir entre chunks con palabras
// semánticamente relacionadas vs. chunks que CONTIENEN las palabras
// exactas de la query.
//
// Para una query como "elementos de juego":
//   - Dense: "COMUNICACIÓN DURANTE EL JUEGO" score=0.749 (gana por "juego")
//   - BM25:  "2. ELEMENTOS DE JUEGO" score=alto (tiene exactamente "elementos" + "juego")
//
// La combinación de ambos via RRF da el resultado correcto.
package bm25

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"strings"
	"unicode"
)

// Parámetros BM25 estándar
const (
	K1 = 1.5  // saturación de frecuencia de términos
	B  = 0.75 // normalización por longitud de documento
)

// Document es un chunk indexado para BM25.
type Document struct {
	ID     string         `json:"id"`
	Text   string         `json:"text"`
	TF     map[string]int `json:"tf"`
	Length int            `json:"length"`
}

// Index es el índice BM25 completo para un conjunto de chunks.
type Index struct {
	Docs   []Document         `json:"docs"`
	AvgLen float64            `json:"avg_len"`
	IDF    map[string]float64 `json:"idf"`
	N      int                `json:"n"`
	DocID  string             `json:"doc_id"`
}

// Result es un chunk con su score BM25 y su ranking original de Qdrant.
type Result struct {
	ID         string
	Text       string
	BM25Score  float64
	DenseScore float32
	DenseRank  int
	BM25Rank   int
	RRFScore   float64
	Source     string
	Page       int
}

// tokenize normaliza y tokeniza texto en español.
// Convierte a minúsculas, elimina acentos (para robustez), filtra stopwords y palabras cortas.
func tokenize(text string) []string {
	// Normalizar a minúsculas
	text = strings.ToLower(text)

	// Normalizar caracteres especiales del español para matching robusto
	replacer := strings.NewReplacer(
		"á", "a", "é", "e", "í", "i", "ó", "o", "ú", "u",
		"ü", "u", "ñ", "n",
	)
	text = replacer.Replace(text)

	// Split en palabras (cualquier no-letra es separador)
	re := regexp.MustCompile(`[^\p{L}0-9]+`)
	words := re.Split(text, -1)

	// Stopwords básicas en español (no agregar valor semántico)
	stopwords := map[string]bool{
		"el": true, "la": true, "los": true, "las": true,
		"un": true, "una": true, "unos": true, "unas": true,
		"de": true, "del": true, "en": true, "con": true,
		"por": true, "para": true, "que": true, "se": true,
		"al": true, "su": true, "sus": true, "es": true,
		"son": true, "ha": true, "han": true, "no": true,
		"si": true, "ya": true, "o": true, "y": true,
		"a": true, "e": true, "u": true,
		"como": true, "cuales": true, "cual": true,
		"este": true, "esta": true, "esto": true,
		"ese": true, "esa": true, "eso": true,
	}

	var tokens []string
	for _, w := range words {
		w = strings.TrimFunc(w, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		})
		if len([]rune(w)) < 3 {
			continue
		}
		if stopwords[w] {
			continue
		}
		tokens = append(tokens, w)
	}
	return tokens
}

// termFreq calcula la frecuencia de cada término en un texto.
func termFreq(text string) (map[string]int, int) {
	tokens := tokenize(text)
	tf := make(map[string]int, len(tokens))
	for _, t := range tokens {
		tf[t]++
	}
	return tf, len(tokens)
}

// Build construye un índice BM25 a partir de una lista de documentos.
// Llamar durante la indexación, después de procesar los chunks.
func Build(docs []struct{ ID, Text string }, docID string) *Index {
	if len(docs) == 0 {
		return &Index{DocID: docID}
	}

	// Construir documentos con TF
	indexDocs := make([]Document, len(docs))
	totalLen := 0
	dfCount := make(map[string]int) // document frequency

	for i, d := range docs {
		tf, length := termFreq(d.Text)
		indexDocs[i] = Document{
			ID:     d.ID,
			Text:   d.Text,
			TF:     tf,
			Length: length,
		}
		totalLen += length
		for term := range tf {
			dfCount[term]++
		}
	}

	avgLen := float64(totalLen) / float64(len(docs))
	n := len(docs)

	// Calcular IDF para cada término
	// IDF = log((N - df + 0.5) / (df + 0.5) + 1)
	idf := make(map[string]float64, len(dfCount))
	for term, df := range dfCount {
		idf[term] = math.Log((float64(n-df)+0.5)/float64(df+1)) + 1
		if idf[term] < 0 {
			idf[term] = 0.01 // Evitar IDF negativo
		}
	}

	return &Index{
		Docs:   indexDocs,
		AvgLen: avgLen,
		IDF:    idf,
		N:      n,
		DocID:  docID,
	}
}

// Score calcula el score BM25 de un documento para una query.
func (idx *Index) Score(docID string, query string) float64 {
	// Encontrar el documento
	var doc *Document
	for i := range idx.Docs {
		if idx.Docs[i].ID == docID {
			doc = &idx.Docs[i]
			break
		}
	}
	if doc == nil {
		return 0
	}

	tokens := tokenize(query)
	if len(tokens) == 0 {
		return 0
	}

	score := 0.0
	docLen := float64(doc.Length)

	for _, term := range tokens {
		tf := float64(doc.TF[term])
		if tf == 0 {
			continue
		}
		idf := idx.IDF[term]
		if idf == 0 {
			idf = 0.1 // término no visto en corpus → IDF bajo
		}

		// Fórmula BM25
		numerator := tf * (K1 + 1)
		denominator := tf + K1*(1-B+B*docLen/idx.AvgLen)
		score += idf * (numerator / denominator)
	}

	return score
}

// SearchAll calcula BM25 score para TODOS los documentos del índice.
// Útil para hacer BM25-first retrieval sobre el corpus completo.
func (idx *Index) SearchAll(query string, topK int) []struct {
	ID    string
	Score float64
} {
	type scored struct {
		ID    string
		Score float64
	}

	scores := make([]scored, 0, len(idx.Docs))
	for _, doc := range idx.Docs {
		s := idx.Score(doc.ID, query)
		if s > 0 {
			scores = append(scores, scored{ID: doc.ID, Score: s})
		}
	}

	// Ordenar por score descendente
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].Score > scores[i].Score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	if topK > len(scores) {
		topK = len(scores)
	}
	result := make([]struct {
		ID    string
		Score float64
	}, topK)
	for i := 0; i < topK; i++ {
		result[i].ID = scores[i].ID
		result[i].Score = scores[i].Score
	}
	return result
}

// RRFScore calcula el score Reciprocal Rank Fusion para un rank dado.
// k=60 es el valor estándar de la literatura.
func RRFScore(rank int) float64 {
	return 1.0 / (60.0 + float64(rank+1))
}

// FuseResults combina resultados de búsqueda densa (Qdrant) con BM25 via RRF.
//
// denseResults: slice de (ID, DenseScore) ordenado por score descendente
// bm25Scores:   map de ID → BM25Score (de todos los docs del índice)
//
// Retorna los resultados combinados ordenados por RRF score descendente.
func FuseResults(
	denseResults []struct {
		ID         string
		DenseScore float32
		Text       string
		Source     string
		Page       int
	},
	idx *Index,
	query string,
	finalK int,
) []Result {
	if idx == nil || len(denseResults) == 0 {
		// Sin índice BM25: devolver resultados densos tal cual
		results := make([]Result, len(denseResults))
		for i, r := range denseResults {
			results[i] = Result{
				ID:         r.ID,
				Text:       r.Text,
				DenseScore: r.DenseScore,
				DenseRank:  i,
				RRFScore:   RRFScore(i),
			}
		}
		return results[:min(finalK, len(results))]
	}

	// Calcular BM25 scores para TODOS los docs del índice
	type bm25scored struct {
		ID    string
		Score float64
	}
	bm25All := make([]bm25scored, 0, len(idx.Docs))
	for _, doc := range idx.Docs {
		s := idx.Score(doc.ID, query)
		bm25All = append(bm25All, bm25scored{ID: doc.ID, Score: s})
	}
	// Ordenar BM25 por score
	for i := 0; i < len(bm25All)-1; i++ {
		for j := i + 1; j < len(bm25All); j++ {
			if bm25All[j].Score > bm25All[i].Score {
				bm25All[i], bm25All[j] = bm25All[j], bm25All[i]
			}
		}
	}
	// Índice de rank BM25 por ID
	bm25Rank := make(map[string]int, len(bm25All))
	bm25Score := make(map[string]float64, len(bm25All))
	for i, b := range bm25All {
		bm25Rank[b.ID] = i
		bm25Score[b.ID] = b.Score
	}

	// Índice de rank denso por ID
	denseRank := make(map[string]int, len(denseResults))
	for i, r := range denseResults {
		denseRank[r.ID] = i
	}

	// Candidatos: unión de resultados densos Y top BM25
	candidates := make(map[string]bool)
	for _, r := range denseResults {
		candidates[r.ID] = true
	}
	// Agregar top-10 BM25 como candidatos adicionales
	for i, b := range bm25All {
		if i >= 10 {
			break
		}
		candidates[b.ID] = true
	}

	// Calcular RRF score para cada candidato
	// RRF(d) = Σ 1/(k + rank(d, list_i)) para cada lista i
	type candidate struct {
		ID         string
		RRFScore   float64
		BM25Score  float64
		DenseScore float32
		DenseRank  int
		BM25Rank   int
	}

	results := make([]candidate, 0, len(candidates))
	for id := range candidates {
		dr, denseOK := denseRank[id]
		br := bm25Rank[id]

		var rrfScore float64
		var ds float32

		if denseOK {
			rrfScore += RRFScore(dr)
			ds = denseResults[dr].DenseScore
		} else {
			rrfScore += RRFScore(len(denseResults)) // penalidad por no estar en dense
		}
		rrfScore += RRFScore(br)

		results = append(results, candidate{
			ID:         id,
			RRFScore:   rrfScore,
			BM25Score:  bm25Score[id],
			DenseScore: ds,
			DenseRank:  dr,
			BM25Rank:   br,
		})
	}

	// Ordenar por RRF score descendente
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].RRFScore > results[i].RRFScore {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Construir resultado final, buscando el texto del chunk
	textByID := make(map[string]struct {
		Text   string
		Source string
		Page   int
	})
	for _, r := range denseResults {
		textByID[r.ID] = struct {
			Text   string
			Source string
			Page   int
		}{r.Text, r.Source, r.Page}
	}
	for _, d := range idx.Docs {
		if _, ok := textByID[d.ID]; !ok {
			textByID[d.ID] = struct {
				Text   string
				Source string
				Page   int
			}{Text: d.Text}
		}
	}

	if finalK > len(results) {
		finalK = len(results)
	}

	final := make([]Result, finalK)
	for i, r := range results[:finalK] {
		info := textByID[r.ID]
		final[i] = Result{
			ID:         r.ID,
			Text:       info.Text,
			BM25Score:  r.BM25Score,
			DenseScore: r.DenseScore,
			DenseRank:  r.DenseRank,
			BM25Rank:   r.BM25Rank,
			RRFScore:   r.RRFScore,
		}
	}

	return final
}

// Save persiste el índice en disco como JSON.
func (idx *Index) Save(path string) error {
	data, err := json.Marshal(idx)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load carga un índice desde disco.
func Load(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var idx Index
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, err
	}
	return &idx, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
