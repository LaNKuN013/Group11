Tenant Chatbot Assistant

A single-file Streamlit app that helps tenants with general chat (offline), contract Q&A (RAG), repair tickets, and rent reminders.
Supports English / 中文 bilingual UI and uses Neon PostgreSQL for storage.

✨ Features
功能 (Feature)	描述 (Description)
💬 General Chat (offline)	No API needed; small-talk assistant with language guard
📄 Contract Q&A (RAG)	Upload PDF tenancy agreement → build FAISS → ask questions
🧰 Repair Tickets	Add / list / delete tickets, stored in Neon DB
💰 Rent Reminders	Add / list / delete reminders, stored in Neon DB
🌐 Bilingual UI	English / 中文 toggle in sidebar
🔧 Diagnostics	Test Neon DB connection + detect API key
🛠 Requirements

Python 3.10+

Neon PostgreSQL database

(Optional) OpenAI API key (OPENAI_API_KEY)

📦 Installation
pip install -r requirements.txt


requirements.txt example:

streamlit==1.38.0
langchain==0.3.0
langchain-core==0.3.0
langchain-community==0.3.0
langchain-openai==0.2.0
openai>=1.47.0
faiss-cpu>=1.8.0
pypdf>=3.17.0
python-dotenv>=1.0.1
psycopg2-binary>=2.9.9
numpy>=1.26.4
pandas>=2.2.3
tqdm>=4.66.5
requests>=2.32.3

🔐 Environment Variables

Set one of these:

# Option A: full Neon URI (recommended)
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DB?sslmode=require"


Or:

# Option B: split fields
export PG_HOST="..."
export PG_DB="..."
export PG_USER="..."
export PG_PASSWORD="..."
export PG_PORT="5432"


Optional (required only for contract Q&A):

export OPENAI_API_KEY="sk-xxxx"

▶️ Run
streamlit run app_tenantbot_neon_single.py


Open browser at http://localhost:8501

🚀 How to Use
1️⃣ General Chat (Offline)

No API key required

Small talk, basic answers

2️⃣ Contract Q&A (RAG)

Sidebar → enter OpenAI API key

Upload PDFs (tenancy agreement / house rules)

Click Build/Refresh Knowledge Base

Ask questions in chat box

3️⃣ Repair Tickets

Enter issue title & description

Each ticket supports:

✅ Add

✅ Delete (top-right ✖)

✅ Save success banner: "Saved! (Total: X)"

4️⃣ Rent Reminders

Choose day of month + note

Each reminder supports:

✅ Add

✅ Delete (top-right ✖)

✅ Save success banner: "Saved! (Total: X)"

🗄 Database Schema (auto-created)
CREATE TABLE IF NOT EXISTS repair_tickets (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'open',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rent_reminders (
  id SERIAL PRIMARY KEY,
  day_of_month INT NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
  note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

🧠 RAG Architecture
Uploaded PDFs → PyPDFLoader → Text Splitter → OpenAIEmbeddings → FAISS
                                 ↓
                           ConversationalRetrievalChain
                                 ↓
                            Semantic Q&A

🧰 Troubleshooting
Issue	Solution
invalid_api_key	Check sidebar → API setup
no such table repair_tickets	Click any feature once; schema auto-creates
Pages reset uploaded PDF	design uses session_state and uploader_key to persist
📄 License

MIT — freely use for learning / academic project / demo.
