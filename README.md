# Estimator-Analytics

A comprehensive multilingual voice and text search system for interior design and furniture estimation, powered by AI and vector search.

## Features

- **Multilingual Voice Search**: Speak in any language (Hindi, Bengali, Tamil, Telugu, Gujarati, Punjabi, Marathi, English)
- **Intelligent Translation**: Automatic language detection and translation to English
- **Vector Search**: Semantic search using Qdrant vector database
- **Precise Filtering**: Keyword-based filtering for accurate results
- **Real-time Processing**: Fast audio-to-text conversion using faster-whisper

## Voice Search

The system supports voice search in multiple Indian languages with automatic translation to English.

### Usage

Send an audio file to the voice search endpoint:

```bash
curl -X POST "http://localhost:8000/search/voice" \
     -F "file=@sample.wav"
```

### Supported Audio Formats

- WAV
- MP3
- M4A
- FLAC
- OGG
- WebM

### Response Format

```json
{
  "query_transcribed": "बेड और कंसोल टेबल",
  "query_translated": "bed and console table",
  "detected_language": "hi",
  "translation_confidence": 0.95,
  "results": [
    {
      "id": 5003,
      "score": 0.87,
      "item_name": "Console Table",
      "attributes_parsed": {
        "Material": "HDHMR",
        "Finish": "PU Paint",
        "Rate": "99 Per sq-ft",
        "Measurement": "69x99 feet"
      },
      "image": "https://.../Console%20Table.png",
      "amount": "676269.00"
    }
  ],
  "total_found": 1
}
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Start the services:
   ```bash
   docker-compose up
   ```

## API Endpoints

### Voice Search
- `POST /search/voice` - Upload audio file for voice search

### Text Search
- `POST /search/vector` - Vector similarity search
- `POST /search/multilingual` - Multilingual text search

### Data Management
- `POST /ingest` - Ingest data into vector database
- `POST /insert/item` - Insert single item
- `POST /insert/items` - Insert multiple items

## Architecture

The system uses a pipeline approach:

1. **Audio Input** → faster-whisper (STT)
2. **Text** → Language Detection → Translation
3. **English Text** → Vector Search (Qdrant)
4. **Results** → Keyword Filtering → Response

## Dependencies

- **STT**: faster-whisper (offline, fast)
- **Translation**: Google Translator + HuggingFace models
- **Vector DB**: Qdrant
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Backend**: FastAPI
- **Frontend**: React + Vite