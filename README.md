# Tenant Chatbot Assistant

A single-file **Streamlit** web app that helps tenants with:

- 💬 General Chat (offline, no API required)
- 📄 Contract Q&A (RAG using LangChain + FAISS + PDF Upload)
- 🧰 Repair Tickets (create / delete persisted in Neon DB)
- 💰 Rent Reminders (create / delete persisted in Neon DB)
- 🌐 English / 中文 bilingual UI
- 🔧 DB & API diagnostics panel

---

## ✨ Features

| Feature | Feature Description |
|------|----------------------|
| 💬 General Chat (offline) | Fast responses, no API needed |
| 📄 Contract Q&A | Upload tenancy agreement PDF → Build FAISS → Ask questions |
| 🧰 Repair Tickets | Tenant submits repair requests; stored in DB |
| 💰 Rent Reminders | Save rent reminder (Day-of-month + note) |
| 🌐 Multi-language UI | English / 中文 toggle from sidebar |
| 🔧 Diagnostics | Test Neon DB connection + detect OpenAI API |

---

## 🔧 Technical Stack

| Component                | Technology Stack                                                            | Purpose / Usage                                                |
| ------------------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Frontend UI**          | Streamlit                                                                   | Web interface (chat, repair tickets, rent reminder management) |
| **Language Model / RAG** | LangChain + OpenAI (optional)                                               | Contract Q&A (PDF → embedding → vector search)                 |
| **Vector Store**         | FAISS (local, via LangChain)                                                | Stores embeddings for similarity search                        |
| **Offline Tenant Chat**  | LangChain `ConversationBufferMemory`                                        | General chat without requiring an API key                      |
| **Database**             | **Neon PostgreSQL (cloud)**                                                 | Stores repair tickets and rent reminders                       |
| **Runtime Environment**  | Python 3.10+                                                                | Execution environment                                          |
| **Dependencies**         | `streamlit`, `langchain`, `openai`, `faiss-cpu`, `psycopg2-binary`, `pypdf` | Required packages for app functionality                        |

---

## 🛠 Requirements

- Python **3.10+**
- Neon PostgreSQL database
- (Optional, for PDF Q&A) OpenAI API key

---

## 📦 Installation

```bash
pip install -r requirements.txt

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
```

---

## 📁 Project Structure
```
📂 Tenant Chatbot Assistant
│
├── app_tenantbot_neon_single.py           # Streamlit UI + logic + DB + RAG)
├── requirements.txt       
├── README.md              
│
├── .env (optional)        # repo：DATABASE_URL / OPENAI_API_KEY
│
└── data/ (runtime generated)
    ├── contract_pdf/      #  PDF (not committed)
    └── vectorstore/       # FAISS index embedding (not committed)
```

---

## 🗄 Database Schema (auto-created)
```sql
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
```

---

## 🧠 RAG Pipeline (Contract Q&A)
```text
PDF Upload → PyPDFLoader → Text Splitter → OpenAI Embeddings → FAISS Vectorstore
               ↓
        ConversationalRetrievalChain
               ↓
           Contract Answer
```

---

## 🚀 Quick Start
```bush
streamlit run app_tenantbot_neon_single.py
```
or open the hosted app:

🔗 https://dss5105group11sasadawdqdd.streamlit.app/


---

## 📄 License

MIT License — free to use & modify.


