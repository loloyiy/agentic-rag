# Agentic RAG System

An intelligent document assistant that handles both unstructured text (PDF, TXT, Word, Markdown) via semantic vector search and structured tabular data (CSV, Excel, JSON) via SQL queries. Built with a Python FastAPI backend and React frontend, styled similar to OpenWebUI.

## Features

- **Document Management**: Upload, organize, and manage documents in collections
- **Multi-Format Support**:
  - Text documents: PDF, TXT, Word (.docx), Markdown
  - Tabular data: CSV, Excel (.xlsx), JSON
- **AI-Powered Chat**: Conversational interface with intelligent tool selection
- **Vector Search**: Semantic search for unstructured text with re-ranking
- **SQL Analysis**: Query structured data using natural language
- **Cross-Document Analysis**: Query and compare multiple documents simultaneously
- **Bilingual Support**: Italian and English responses
- **Export & Backup**: Export analysis results and create full backups

## Technology Stack

### Backend
- **Runtime**: Python 3.11+
- **Framework**: FastAPI
- **AI**: LangChain / LangGraph
- **Database**: PostgreSQL with pgvector extension
- **LLM**: OpenAI (GPT-4o, GPT-4o-mini) or Ollama
- **Embeddings**: OpenAI text-embedding-3-small or Ollama
- **Re-ranking**: Cohere or Cross-Encoder

### Frontend
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand + React Query
- **UI Style**: OpenWebUI-inspired interface

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 or higher
- Node.js 18 or higher
- PostgreSQL 15+ with pgvector extension
- OpenAI API key (configurable via UI)
- (Optional) Ollama for local model support
- (Optional) Cohere API key for re-ranking

## Quick Start

1. **Clone the repository**
   ```bash
   cd RAG_4
   ```

2. **Run the setup script**
   ```bash
   ./init.sh
   ```

   This script will:
   - Check prerequisites
   - Set up Python virtual environment
   - Install backend dependencies
   - Install frontend dependencies
   - Start both servers

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

4. **Configure API keys**
   - Open Settings from the sidebar
   - Enter your OpenAI API key
   - (Optional) Enter Cohere API key for re-ranking
   - Select your preferred models

## Project Structure

```
RAG_4/
├── backend/
│   ├── api/              # API route handlers
│   ├── core/             # Core application logic
│   ├── models/           # Database models
│   ├── services/         # Business logic services
│   ├── utils/            # Utility functions
│   ├── main.py           # FastAPI entry point
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── services/     # API service functions
│   │   ├── styles/       # CSS styles
│   │   └── utils/        # Utility functions
│   ├── public/           # Static assets
│   └── package.json      # Node dependencies
├── prompts/              # AI prompts and configurations
├── init.sh               # Environment setup script
└── README.md             # This file
```

## Database Setup

The application uses PostgreSQL with the pgvector extension. To set up the database:

1. Install PostgreSQL 15+
2. Install the pgvector extension:
   ```sql
   CREATE EXTENSION vector;
   ```
3. Create a database:
   ```sql
   CREATE DATABASE agentic_rag;
   ```
4. Configure the connection in `backend/.env`

## Development

### Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm run dev
```

## API Endpoints

### Documents
- `POST /api/ingest` - Upload and process document
- `GET /api/documents` - List all documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document

### Collections
- `POST /api/collections` - Create collection
- `GET /api/collections` - List collections
- `DELETE /api/collections/{id}` - Delete collection

### Chat
- `POST /api/chat` - Send message and get response
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation

### Settings
- `GET /api/settings` - Get configuration
- `PATCH /api/settings` - Update configuration

## License

MIT License
