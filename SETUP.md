# 🚀 Setup Guide - Fireworks AI Enhanced Playground

This guide will help you get the Fireworks AI Enhanced Playground up and running in minutes.

## Prerequisites

- Docker and Docker Compose installed
- API keys for Fireworks AI (required)
- Optional: DeepInfra API key, SerpAPI key

## Quick Start (5 minutes)

### Step 1: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/groxaxo/fireworks.git
cd fireworks

# Copy environment template
cp .env.example .env
```

### Step 2: Add Your API Keys

Edit `.env` and add your API keys:

```env
FIREWORKS_API_KEY=your_fireworks_key_here
DEEPINFRA_API_KEY=your_deepinfra_key_here  # Optional
SERPAPI_KEY=your_serpapi_key_here           # Optional
```

### Step 3: Start with Docker

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f app
```

### Step 4: Access the Application

Open your browser and navigate to:
- **Streamlit UI**: http://localhost:8501
- **ChromaDB Admin**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend                      │
│                  Port: 8501                              │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────────────────┐
         │                                   │
┌────────▼────────┐              ┌──────────▼─────────┐
│  PostgreSQL     │              │    ChromaDB        │
│  (pgvector)     │              │  Port: 8000        │
│  Port: 5432     │              └────────────────────┘
└─────────────────┘
```

## Component Details

### 1. Streamlit Application
- **Purpose**: User interface for AI interactions
- **Features**: 
  - Multi-provider model selection
  - Web search integration
  - Deep search capabilities
  - Chat history and analytics
  - MCP context management

### 2. PostgreSQL with pgvector
- **Purpose**: Store embeddings and search history
- **Tables**:
  - `embeddings`: Vector storage with similarity search
  - `search_history`: Query and response history
  - `deep_search_results`: Deep search findings

### 3. ChromaDB
- **Purpose**: Specialized vector database for findings
- **Use Cases**:
  - Semantic search across findings
  - RAG (Retrieval Augmented Generation)
  - Document similarity

## Features Breakdown

### 🤖 Multi-Provider Support

**Fireworks AI**:
- 100+ text models (Llama, Mixtral, Gemma, etc.)
- Image generation (Stable Diffusion XL, Playground v2)
- Fast inference with optimized infrastructure

**DeepInfra**:
- 24+ models including Llama, Qwen, Mistral
- Vision models (Llama 3.2 Vision)
- Embedding models (BGE, E5, GTE)

### 🔍 Search Capabilities

**Web Search**:
- Real-time web search via SerpAPI
- Context augmentation for responses
- Source attribution

**Deep Search**:
- Multi-level recursive search
- Related query exploration
- Aggregated findings across levels

### 🧠 Advanced AI Features

**Embeddings**:
- Generate embeddings for text
- Store in PostgreSQL with pgvector
- Semantic similarity search

**Reranking**:
- Cosine similarity reranking
- Reciprocal Rank Fusion (RRF)
- Hybrid scoring (semantic + lexical + position)

**MCP (Model Context Protocol)**:
- Standardized context management
- Cross-provider compatibility
- Conversation state preservation

## API Key Setup

### Fireworks AI (Required)

1. Visit https://fireworks.ai/
2. Sign up for a free account
3. Navigate to https://fireworks.ai/api-keys
4. Create a new API key
5. Add to `.env` as `FIREWORKS_API_KEY`

**Free Tier**: Generous free tier available

### DeepInfra (Optional)

1. Visit https://deepinfra.com/
2. Sign up for an account
3. Go to Settings → API Tokens
4. Create a new token
5. Add to `.env` as `DEEPINFRA_API_KEY`

**Free Tier**: $1 free credit on signup

### SerpAPI (Optional)

1. Visit https://serpapi.com/
2. Sign up for an account
3. Go to Dashboard → API Key
4. Copy your API key
5. Add to `.env` as `SERPAPI_KEY`

**Free Tier**: 100 searches/month

## Usage Examples

### Example 1: Basic Text Generation

1. Select "Fireworks AI" provider
2. Choose a text model (e.g., Llama 3.1 70B)
3. Enter prompt: "Explain quantum computing"
4. Click "Generate"

### Example 2: Web-Enhanced Generation

1. Enable "Web Search" in sidebar
2. Enter prompt: "What are the latest developments in AI?"
3. System will:
   - Search web for current information
   - Include results in context
   - Generate informed response

### Example 3: Deep Search

1. Enable "Deep Search" in sidebar
2. Set depth to 3
3. Enter: "Best practices for Docker deployment"
4. System will:
   - Perform 3 levels of recursive search
   - Follow related queries
   - Aggregate and present findings

### Example 4: Image Generation

1. Select "Image" model type
2. Choose "Stable Diffusion XL"
3. Enter: "A futuristic city at sunset"
4. Click "Generate"

## Troubleshooting

### Services Won't Start

```bash
# Check Docker daemon
docker info

# Check ports are available
lsof -i :8501
lsof -i :5432
lsof -i :8000

# Restart services
docker compose down
docker compose up -d
```

### Database Connection Issues

```bash
# Check PostgreSQL logs
docker compose logs postgres

# Verify database is ready
docker compose exec postgres pg_isready

# Recreate database
docker compose down -v
docker compose up -d
```

### ChromaDB Issues

```bash
# Check ChromaDB logs
docker compose logs chromadb

# Verify ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# Recreate ChromaDB
docker compose down -v
docker compose up -d
```

### Application Errors

```bash
# Check app logs
docker compose logs app

# Restart app only
docker compose restart app

# Rebuild app
docker compose up -d --build app
```

## Local Development (Without Docker)

If you prefer to run locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL (requires Docker)
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=fireworks_user \
  -e POSTGRES_PASSWORD=fireworks_password \
  -e POSTGRES_DB=fireworks_db \
  ankane/pgvector:latest

# Start ChromaDB (requires Docker)
docker run -d -p 8000:8000 chromadb/chroma:latest

# Set environment variables
export FIREWORKS_API_KEY=your_key
export POSTGRES_HOST=localhost
export CHROMA_HOST=localhost

# Run Streamlit
streamlit run streamlit_app.py
```

## Performance Optimization

### For Production

1. **Enable Redis caching** (add to docker-compose.yml)
2. **Scale horizontally** with load balancer
3. **Use persistent volumes** for data
4. **Enable monitoring** with Prometheus/Grafana
5. **Configure resource limits** in docker-compose.yml

### Resource Limits Example

```yaml
services:
  app:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Security Best Practices

1. **Never commit `.env`** with real API keys
2. **Use secrets management** in production (AWS Secrets Manager, etc.)
3. **Enable HTTPS** with reverse proxy (nginx, Caddy)
4. **Restrict database access** with firewall rules
5. **Regular updates** of dependencies and images

## Monitoring

### Health Checks

```bash
# Application health
curl http://localhost:8501/_stcore/health

# ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# PostgreSQL health
docker compose exec postgres pg_isready
```

### Logs

```bash
# View all logs
docker compose logs -f

# View specific service
docker compose logs -f app
docker compose logs -f postgres
docker compose logs -f chromadb
```

## Backup and Restore

### Database Backup

```bash
# Backup PostgreSQL
docker compose exec postgres pg_dump -U fireworks_user fireworks_db > backup.sql

# Restore PostgreSQL
cat backup.sql | docker compose exec -T postgres psql -U fireworks_user fireworks_db
```

### ChromaDB Backup

```bash
# Backup ChromaDB data
docker compose exec chromadb tar czf /backup/chroma.tar.gz /chroma/chroma

# Copy backup out
docker compose cp chromadb:/backup/chroma.tar.gz ./chroma_backup.tar.gz
```

## Next Steps

1. **Explore the UI**: Try different models and features
2. **Customize**: Modify `streamlit_app.py` for your needs
3. **Extend**: Add new services in `services/` directory
4. **Deploy**: Push to cloud provider of choice
5. **Monitor**: Set up logging and monitoring

## Support

- **GitHub Issues**: https://github.com/groxaxo/fireworks/issues
- **Documentation**: See README.md
- **Fireworks Docs**: https://docs.fireworks.ai/

---

Happy coding! 🎆
