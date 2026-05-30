// Package cache implementa un cache persistente en disco para embeddings.
// Usa bbolt (BoltDB) como motor de almacenamiento key-value embebido.
//
// La clave es el SHA256 del texto del chunk. Si el texto no cambió,
// el embedding tampoco cambia, por lo que el hash es un identificador perfecto.
// Esto hace que re-indexar el mismo PDF sea casi instantáneo.
package cache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"

	bolt "go.etcd.io/bbolt"
)

const bucketName = "embeddings"
const bboltMode = 0600

// Cache maneja el almacenamiento persistente de embeddings en disco.
type Cache struct {
	db *bolt.DB
}

// New abre o crea la base de datos de cache en el path especificado.
func New(path string) (*Cache, error) {
	// Crear el directorio si no existe
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, fmt.Errorf("crear directorio cache: %w", err)
	}

	// Abrir o crear la base de datos bbolt en el path dado. El modo 0600 asegura que solo el usuario actual pueda leer/escribir el archivo.
	db, err := bolt.Open(path, bboltMode, nil)
	if err != nil {
		return nil, fmt.Errorf("abrir cache: %w", err)
	}

	// Crear el bucket si no existe
	err = db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(bucketName))
		return err
	})
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("crear bucket: %w", err)
	}

	return &Cache{db: db}, nil
}

// HashText genera el SHA256 de un texto. Se usa como clave de cache.
func HashText(text string) string {
	// Calcular el hash SHA256 del texto y devolverlo como string hexadecimal.
	h := sha256.Sum256([]byte(text))
	// hex.EncodeToString convierte el hash a una representación legible en hexadecimal.
	return hex.EncodeToString(h[:])
}

// Get recupera un embedding del cache. Devuelve (embedding, true) si existe,
// o (nil, false) si no está cacheado.
func (c *Cache) Get(hash string) ([]float32, bool) {
	var result []float32

	// Usamos View porque solo necesitamos leer el embedding, no modificar nada.
	c.db.View(func(tx *bolt.Tx) error {
		// Obtener el bucket y luego el valor asociado a la clave hash usando Get.
		b := tx.Bucket([]byte(bucketName))
		// Get devuelve nil si la clave no existe, así que verificamos eso para saber si el embedding está cacheado.
		v := b.Get([]byte(hash))
		if v == nil {
			return nil
		}
		// Si el valor existe, lo convertimos de bytes a []float32 usando bytesToFloat32Slice.
		result = bytesToFloat32Slice(v)
		return nil
	})

	return result, result != nil
}

// Set guarda un embedding en el cache con su hash como clave.
func (c *Cache) Set(hash string, embedding []float32) error {
	// Usamos Update porque vamos a modificar el bucket guardando un nuevo embedding.
	return c.db.Update(func(tx *bolt.Tx) error {
		// Obtener el bucket y guardar el embedding serializado como bytes usando Put.
		b := tx.Bucket([]byte(bucketName))
		// float32SliceToBytes convierte el slice de float32 a []byte para almacenarlo en bbolt.
		return b.Put([]byte(hash), float32SliceToBytes(embedding))
	})
}

// Stats devuelve cuántos embeddings hay guardados en el cache.
func (c *Cache) Stats() (int, error) {
	var count int
	// Usamos View porque solo necesitamos leer las estadísticas del bucket, no modificar nada.
	err := c.db.View(func(tx *bolt.Tx) error {
		// Obtener el bucket y contar cuántas claves tiene usando Stats().KeyN.
		b := tx.Bucket([]byte(bucketName))
		// Stats() devuelve un struct con varias estadísticas, y KeyN es el número de claves en el bucket.
		count = b.Stats().KeyN
		return nil
	})
	return count, err
}

// Close cierra la base de datos limpiamente.
func (c *Cache) Close() error {
	return c.db.Close()
}

// --- Serialización ---

// float32SliceToBytes convierte []float32 a []byte para guardar en bbolt.
// Usa little-endian para compatibilidad.
func float32SliceToBytes(floats []float32) []byte {
	// Cada float32 ocupa 4 bytes, así que el buffer es 4 veces el tamaño del slice.
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		// Convertir el float32 a uint32 bits y escribirlo en el buffer
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// bytesToFloat32Slice convierte []byte de vuelta a []float32.
func bytesToFloat32Slice(buf []byte) []float32 {
	// El buffer debe ser múltiplo de 4 bytes para ser válido
	if len(buf)%4 != 0 {
		return nil
	}
	// Cada 4 bytes representan un float32, así que el slice de floats es un cuarto del tamaño del buffer.
	floats := make([]float32, len(buf)/4)
	// Iterar sobre el buffer de 4 en 4 bytes para reconstruir los float32
	for i := range floats {
		// Leer 4 bytes del buffer, convertirlos a uint32 bits y luego a float32
		bits := binary.LittleEndian.Uint32(buf[i*4:])
		// Convertir los bits a float32 y asignar al slice
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}
