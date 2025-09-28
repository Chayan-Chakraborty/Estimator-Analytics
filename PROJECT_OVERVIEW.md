# Estimator Analytics - Project Overview

## Project Description

Estimator Analytics is a sophisticated search and analytics platform designed for construction and interior design estimation. The system enables users to search through a database of construction items, furniture, and materials using natural language queries in multiple Indian languages. It provides intelligent search capabilities with multilingual support, voice recognition, and advanced filtering options to help users find relevant items with pricing and measurement data.

## Tech Stack

### Backend
- **Python 3.11** - Core programming language
- **FastAPI** - Modern, fast web framework for building APIs
- **Uvicorn** - ASGI server for running FastAPI applications
- **Qdrant** - Vector database for semantic search and similarity matching
- **MySQL** - Relational database for storing item data
- **Sentence Transformers** - Machine learning models for text embeddings
- **Deep Translator** - Translation services for multilingual support
- **Langdetect** - Language detection for automatic language identification
- **Indic Transliteration** - Support for Indian language transliteration

### Frontend
- **React 18** - Modern JavaScript library for building user interfaces
- **Vite** - Fast build tool and development server
- **Axios** - HTTP client for API communication
- **React Speech to Text** - Voice recognition capabilities
- **Lucide React** - Icon library

### Infrastructure
- **Docker** - Containerization for consistent deployment
- **Docker Compose** - Multi-container orchestration
- **Node.js 20** - JavaScript runtime for frontend development

## Features

### Core Search Capabilities
- **Semantic Search** - Vector-based similarity search using sentence transformers
- **NLP-Enhanced Search** - Natural language processing for intelligent query understanding
- **Multilingual Support** - Search in 7 Indian languages (Bengali, Hindi, Tamil, Telugu, Gujarati, Punjabi, Marathi)
- **Voice Search** - Speech-to-text input with automatic language detection
- **Advanced Filtering** - Filter by item name, city, measurement, and other attributes

### User Interface Features
- **Responsive Design** - Works on desktop and mobile devices
- **Card and Table Views** - Multiple display options for search results
- **Real-time Search** - Instant search results with pagination
- **Interactive Filters** - Sidebar with advanced filtering options
- **Item Details Panel** - Detailed view with statistics and pricing information

### Analytics and Insights
- **Price Analytics** - Min, max, and average pricing by area and item type
- **Measurement Analysis** - Square footage calculations and comparisons
- **Area Statistics** - Location-based pricing insights
- **Attribute Parsing** - Intelligent extraction of material, finish, and measurement data

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   Data Layer    │
│   (React/Vite)   │◄──►│   (FastAPI)      │◄──►│   (Qdrant +     │
│                 │    │                 │    │    MySQL)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Input   │    │   Translation   │    │   Vector Store  │
│   Recognition   │    │   Services      │    │   (Embeddings)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Interaction Flow

1. **User Input** → Frontend captures text/voice input
2. **Language Detection** → Automatic detection of Indian languages
3. **Translation** → Query translated to English for processing
4. **Vector Encoding** → Text converted to embeddings using sentence transformers
5. **Search Execution** → Qdrant performs similarity search
6. **Result Processing** → Backend processes and enriches results
7. **Response** → Frontend displays formatted results with analytics

### Data Flow

```
MySQL Database → Data Ingestion → Qdrant Vector Store
                     ↓
User Query → Language Detection → Translation → Vector Search
                     ↓
Results ← Processing ← Similarity Matching ← Vector Store
```

## Setup & Running Instructions

### Prerequisites
- Docker and Docker Compose installed
- Environment variables configured (`.env` file)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd Estimator-Analytics

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# Qdrant: http://localhost:6333
```

### Environment Variables
Create a `.env` file with the following variables:
```
DB_HOST=localhost
DB_PORT=3306
DB_NAME=vishanti
DB_USER=root
DB_PASS=password
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### Manual Setup (Development)
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## API & Data Flow

### Key API Endpoints

#### Search Endpoints
- `POST /search` - Basic vector search
- `POST /search/nlp` - NLP-enhanced search with filtering
- `POST /search/multilingual` - Multilingual search with translation
- `POST /search/multilingual/nlp` - Multilingual NLP search
- `POST /search/multilingual/vector` - Multilingual vector search

#### Data Management
- `POST /ingest` - Ingest data from MySQL to Qdrant
- `GET /db/items` - Retrieve items from MySQL database
- `POST /insert/item` - Insert single item
- `POST /insert/items` - Insert multiple items

#### Analytics
- `POST /search/filtered-stats` - Get filtered statistics
- `POST /search/filtered-stats/only` - Get area statistics only

### Data Models

#### Core Item Structure
```json
{
  "id": "item_id",
  "item_name": "string",
  "room_name": "string", 
  "project_name": "string",
  "area": "string",
  "amount": "number",
  "attributes": "object",
  "attributes_parsed": "object",
  "measurement_sqft": "number",
  "image": "string"
}
```

#### Multilingual Search Request
```json
{
  "query": "string",
  "source_language": "string",
  "target_language": "string",
  "use_nlp": "boolean",
  "page": "number",
  "page_size": "number"
}
```

### Database Schema
- **MySQL**: Stores relational data (items, rooms, projects, addresses)
- **Qdrant**: Stores vector embeddings and metadata for semantic search
- **Vector Size**: 384 dimensions (all-MiniLM-L6-v2 model)

## Future Improvements / Roadmap

### Immediate Enhancements
- **Enhanced Language Support**: Add more Indian languages and dialects
- **Improved Voice Recognition**: Better accuracy for Indian language speech
- **Advanced Analytics**: More sophisticated pricing and trend analysis
- **Mobile App**: Native mobile application for field use

### Technical Improvements
- **Caching Layer**: Redis integration for improved performance
- **API Rate Limiting**: Implement proper rate limiting and authentication
- **Monitoring**: Add logging, metrics, and health checks
- **Testing**: Comprehensive unit and integration tests

### Feature Additions
- **Recommendation Engine**: AI-powered item recommendations
- **Price Prediction**: Machine learning models for cost estimation
- **Image Search**: Visual similarity search for items
- **Collaboration Features**: Multi-user support and sharing capabilities
- **Export Functionality**: PDF reports and data export options

### Scalability
- **Microservices**: Break down into smaller, focused services
- **Load Balancing**: Handle increased traffic and concurrent users
- **Database Optimization**: Query optimization and indexing improvements
- **CDN Integration**: Faster asset delivery and global accessibility

---

*This project represents a modern approach to construction estimation with AI-powered search capabilities and multilingual support, specifically designed for the Indian market.*

## Tried Solutions (Translation Layer)

- GoogleTranslator + language detection: baseline auto-detect and translate.
- Indic transliteration candidates + scoring: generate multiple script candidates for romanized input and pick the most fluent English.
- NLLB-200 (transformers) via `translate_to_english.py`: robust semantic translation for mixed-language (Hindi/Bengali + English) phrases; now the primary path, with GoogleTranslator as fallback.
- Removed rigid word mappings; the system now relies on AI models for context-aware translation.
