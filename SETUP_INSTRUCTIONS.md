# RAG_4 - Istruzioni di Configurazione su Nuovo Computer

## 📋 Contenuto del Backup

Questo backup contiene:
- ✅ Codice sorgente completo (backend + frontend)
- ✅ Database PostgreSQL (database_backup.sql)
- ✅ File di configurazione (.env)
- ✅ Documentazione del progetto

## 🚀 Procedura di Configurazione

### 1️⃣ Prerequisiti
Installa sul nuovo computer:
- Python 3.11+
- Node.js 18+ e npm
- PostgreSQL 15+ con estensione pgvector
- Git (opzionale)

### 2️⃣ Copia la Cartella
```bash
# Copia RAG_4_Backup in una posizione permanente
cp -r /path/to/RAG_4_Backup ~/Dev/RAG_4
cd ~/Dev/RAG_4
```

### 3️⃣ Installa Dipendenze Python
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4️⃣ Installa Dipendenze Node.js
```bash
cd ../frontend
npm install
```

### 5️⃣ Configura il Database
```bash
# Se il database non esiste
createdb -U postgres agentic_rag

# Ripristina il backup
psql -U postgres agentic_rag < database_backup.sql
```

### 6️⃣ Configura le Variabili di Ambiente
```bash
cd backend
nano .env
# Modifica i valori se necessario:
# - OPENAI_API_KEY (se hai una chiave)
# - DATABASE_URL (se usi credenziali diverse)
# - Altre configurazioni
```

### 7️⃣ Avvia il Progetto

**Terminal 1 - Backend:**
```bash
cd backend
source .venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### 8️⃣ Verifica
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

## 📝 Note Importanti

### Credenziali Database
- User: `postgres`
- Password: `postgres` (di default, cambiarla in produzione)
- Database: `agentic_rag`

### API Keys
Se usi OpenAI o Cohere, inserisci le chiavi in `.env`:
```
OPENAI_API_KEY=sk-your-key-here
COHERE_API_KEY=your-key-here
```

### PostgreSQL con pgvector
Se PostgreSQL non ha pgvector installato:
```bash
sudo -u postgres psql
postgres=# CREATE EXTENSION IF NOT EXISTS vector;
```

## 🆘 Troubleshooting

### "Database does not exist"
```bash
createdb -U postgres agentic_rag
psql -U postgres agentic_rag < database_backup.sql
```

### "Permission denied" su Linux
```bash
sudo chown -R $USER:$USER .
```

### Python dependencies issues
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

### PostgreSQL connection error
Verifica che PostgreSQL sia in esecuzione:
```bash
# macOS
brew services start postgresql

# Linux
sudo systemctl start postgresql

# Windows
# Usa Services Manager per avviare PostgreSQL
```

## 📦 Backup Supplementari
- Tutti gli upload e i backups del database sono salvati separatamente
- Per sincronizzare gli upload: copia la cartella `backend/uploads/` se presente nel backup

---
**Data Backup:** 2026-02-13
**Versione Progetto:** v1.0.1
