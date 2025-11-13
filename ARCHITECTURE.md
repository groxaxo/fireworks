# Architecture Documentation

## System Overview

The Fireworks AI Enhanced Playground is a microservices-based application that combines multiple AI providers with advanced search and storage capabilities.

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                            │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Streamlit Frontend (Port 8501)                 │  │
│  │  • Multi-provider model selection                           │  │
│  │  • Web search interface                                     │  │
│  │  • Chat history & analytics                                 │  │
│  │  • MCP context management                                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────────────────┐
│                      SERVICE LAYER                                │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Fireworks  │  │  DeepInfra   │  │  Web Search  │           │
│  │   Service    │  │   Service    │  │   Service    │           │
│  │  (100+ models)│ │  (24+ models) │ │   (SerpAPI)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │     MCP      │  │   Reranker   │  │   Database   │           │
│  │   Service    │  │   Service    │  │   Service    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────┴──────────────────────────────────────┐
│                      STORAGE LAYER                                │
│                                                                    │
│  ┌─────────────────────────────┐  ┌─────────────────────────┐   │
│  │   PostgreSQL (pgvector)     │  │       ChromaDB          │   │
│  │      Port 5432              │  │       Port 8000         │   │
│  │                             │  │                         │   │
│  │  • embeddings               │  │  • findings storage     │   │
│  │  • search_history           │  │  • semantic search      │   │
│  │  • deep_search_results      │  │  • vector operations    │   │
│  └─────────────────────────────┘  └─────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Text Generation Flow

```
User Input
    ↓
Streamlit UI
    ↓
[Optional] Web Search Service
    ↓
MCP Service (Context Management)
    ↓
Provider Selection (Fireworks/DeepInfra)
    ↓
Model Inference
    ↓
[Optional] Store in Database
    ↓
[Optional] Store in ChromaDB
    ↓
Display Result
```

### 2. Deep Search Flow

```
User Query
    ↓
Web Search Service
    ↓
Level 1 Search → Extract Related Queries
    ↓
Level 2 Search → Extract Related Queries
    ↓
Level N Search
    ↓
Aggregate Results
    ↓
Reranker Service
    ↓
Database Service (Store Results)
    ↓
Display Ranked Results
```

### 3. Embedding & Semantic Search Flow

```
Text Input
    ↓
DeepInfra Embedding Model
    ↓
Generate Embedding Vector
    ↓
Store in PostgreSQL (pgvector)
    ↓
Store in ChromaDB
    ↓
Query Embedding Generated
    ↓
Similarity Search (Cosine Distance)
    ↓
Reranker Service
    ↓
Return Top K Results
```

## Service Details

### 1. DeepInfra Service

**Purpose**: Interface with DeepInfra API for model inference

**Capabilities**:
- Chat completion (17 text models)
- Vision inference (2 vision models)
- Embedding generation (5 embedding models)

**Models**:
- Text: Llama 3.1, Qwen 2.5, Mistral, DeepSeek Coder
- Vision: Llama 3.2 Vision (90B, 11B)
- Embeddings: BGE, GTE, E5, Sentence Transformers

**API**: OpenAI-compatible REST API

### 2. Web Search Service

**Purpose**: Perform web searches and deep searches

**Features**:
- Standard web search (via SerpAPI)
- Deep search with recursive query expansion
- Result extraction and formatting
- Related query discovery

**Deep Search Algorithm**:
1. Start with initial query
2. Search and extract top results
3. Extract related searches from results
4. Queue related searches for next level
5. Repeat until depth reached
6. Aggregate and deduplicate results

### 3. Database Service

**Purpose**: Manage PostgreSQL operations with pgvector

**Tables**:
- `embeddings`: Vector storage with similarity search
- `search_history`: Query and response logging
- `deep_search_results`: Deep search findings

**Operations**:
- Store embeddings with metadata
- Similarity search using cosine distance
- CRUD operations for search history
- Batch operations for efficiency

### 4. ChromaDB Service

**Purpose**: Vector database for semantic search

**Features**:
- Collection management
- Batch operations
- Similarity search
- Metadata filtering
- Document CRUD

**Use Cases**:
- RAG (Retrieval Augmented Generation)
- Semantic search across documents
- Finding similar content

### 5. MCP Service

**Purpose**: Model Context Protocol implementation

**Features**:
- Standardized context management
- Multi-turn conversation support
- Provider-agnostic formatting
- Context export/import
- Message history

**Context Structure**:
```python
{
    "system_prompt": "...",
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "max_tokens": 2048,
    "temperature": 0.7
}
```

### 6. Reranker Service

**Purpose**: Improve search result relevance

**Strategies**:
1. **Cosine Similarity**: Use embedding similarity
2. **Reciprocal Rank Fusion**: Combine multiple rankings
3. **Hybrid Scoring**: Semantic + Lexical + Position

**Algorithm** (Hybrid):
```
score = w1 * semantic_similarity 
      + w2 * keyword_overlap 
      + w3 * (1 / position)
```

## Database Schema

### PostgreSQL Tables

#### embeddings
```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX ON embedding USING ivfflat (embedding vector_cosine_ops)
);
```

#### search_history
```sql
CREATE TABLE search_history (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    results JSONB,
    model_used VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### deep_search_results
```sql
CREATE TABLE deep_search_results (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    search_depth INTEGER,
    results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### ChromaDB Collections

#### fireworks_findings
```python
{
    "name": "fireworks_findings",
    "metadata": {
        "description": "Storage for search findings and results"
    },
    "documents": ["text content"],
    "embeddings": [[float...]],
    "metadatas": [{"key": "value"}]
}
```

## API Endpoints

### Fireworks AI
- Base URL: `https://api.fireworks.ai`
- Authentication: Bearer token
- Models: Text + Image generation

### DeepInfra
- Base URL: `https://api.deepinfra.com/v1/openai`
- Authentication: Bearer token
- Models: Text + Vision + Embeddings

### SerpAPI
- Base URL: `https://serpapi.com/search`
- Authentication: API key parameter
- Engine: Google Search

## Security Considerations

1. **API Keys**: Stored in environment variables, never in code
2. **Database**: Credentials in environment, connection pooling
3. **Network**: Container isolation, no exposed ports except necessary
4. **Input Validation**: Sanitization in all user inputs
5. **Rate Limiting**: Implemented at provider level

## Scalability

### Horizontal Scaling
- Stateless Streamlit app can be replicated
- Load balancer distributes traffic
- Shared database and ChromaDB instances

### Vertical Scaling
- Increase container resources
- Optimize database indexes
- Use caching layer (Redis)

### Performance Optimization
- Connection pooling for databases
- Batch operations where possible
- Async operations for I/O
- Result caching

## Monitoring

### Health Checks
- Streamlit: `/_stcore/health`
- ChromaDB: `/api/v1/heartbeat`
- PostgreSQL: `pg_isready`

### Metrics
- Request count and latency
- Model inference time
- Database query performance
- Search success rate

## Deployment Options

1. **Docker Compose** (Development)
   - Single machine
   - Easy setup
   - Good for testing

2. **Kubernetes** (Production)
   - Multi-node cluster
   - Auto-scaling
   - High availability

3. **Cloud Platforms**
   - AWS ECS/EKS
   - Google Cloud Run
   - Azure Container Instances

## Future Enhancements

1. **Authentication**: User authentication and authorization
2. **Caching**: Redis for response caching
3. **Monitoring**: Prometheus + Grafana dashboard
4. **CI/CD**: Automated testing and deployment
5. **API Gateway**: Rate limiting and routing
6. **Multi-tenancy**: Support multiple users
7. **Advanced RAG**: Document ingestion and indexing
8. **Fine-tuning**: Model customization support

## Technology Stack

- **Frontend**: Streamlit 1.37.1
- **Backend**: Python 3.11
- **Databases**: 
  - PostgreSQL 16 with pgvector
  - ChromaDB 0.4.22
- **AI Providers**:
  - Fireworks AI
  - DeepInfra
- **Search**: SerpAPI
- **Containerization**: Docker, Docker Compose
- **Libraries**: 
  - requests 2.31.0
  - psycopg2-binary 2.9.9
  - numpy 1.26.4
  - python-dotenv 1.0.1

---

Last Updated: 2025-11-12
