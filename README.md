# DSS5105 Capstone Project - Track B: Conversational AI Assistant (Tenant Chatbot)

### Industry Context
Real estate is heavily transaction- and relationship-driven. Landlords, agents, and tenants face information overload, manual back-and-forth communication, and fragmented systems. Generative AI can automate these workflows if applied to the right use cases.

### Problem Statement
Landlords and property managers spend significant time answering repetitive questions from tenants (rent due dates, maintenance responsibilities, contract clauses). Current solutions (WhatsApp/email/manual calls) are time-consuming and inconsistent.

💡Goal: Build a real estate AI system (chatbot) for property inquiries and tenant services. Cloud deployed, with API/UI.

---

## 🤖Tenant Chatbot Assistant

### Homepage
<img width="1439" height="686" alt="homepage.png" src="https://github.com/LaNKuN013/group11/blob/main/READMEimages/homepage.png" />

### Chat Example
<img width="1428" height="685" alt="chatexample.png" src="https://github.com/LaNKuN013/group11/blob/main/READMEimages/chatexample.png" />

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
├── app_tenantbot_neon_single.py        # ✅ Main Streamlit app (UI + logic + DB + RAG in one file)
├── requirements.txt                    # ✅ Python dependencies (required by Streamlit Cloud)
├── README.md                           # ✅ Project documentation
│
├── .env                                # (optional) Environment variables for local development
│                                        #   - OPENAI_API_KEY
│                                        #   - DATABASE_URL (Neon / PostgreSQL)
│
├── .streamlit/                         # ✅ Streamlit configuration folder
│   └── config.toml                     # ✅ UI theme configuration (forces Light mode, prevents black text)
│
└── data/                               # Runtime generated folders (DO NOT commit to GitHub)
    ├── contract_pdf/                   # Temporary folder for uploaded PDFs
    └── vectorstore/                    # FAISS vector index generated from embeddings for RAG
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

### 🏗️ Local

1. Create venv & activate

macOS/Linux
   
```bush
python3 -m venv .venv
source .venv/bin/activate
```

Windows
   
```bush
py -3 -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies

```bush
pip install -r requirements.txt
```
3. Configure .env (Environment Variables)

    Create a file named .env in the project root directory:
   
```bush
# Required for Contract Chat (RAG)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Neon / PostgreSQL connection string (example)
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME
```

You may skip .env if you prefer — you can also enter the OpenAI API Key directly in the sidebar UI. 
When entered via the sidebar, the key is valid for the current session only.

4. Run the app
```bush
streamlit run app_tenantbot_neon_single.py
```

### 💻 Online
Open the hosted app:

🔗 https://dss5105group11sasadawdqdd.streamlit.app/

---

## 🧭 How to Use

#### Language selection
Choose **English / 中文** in the sidebar.

#### General Chat (Offline)
- Does **not** require an API key
- Good for small talk and simple questions

#### Contract Chat (RAG)
1. Enter your **OpenAI API Key** in the sidebar (or configure it in `.env`)
2. Upload tenancy agreement or house rules **PDF**
3. Click **Build/Refresh Knowledge Base** to index the document
4. Ask your question in the input box, e.g., *“What’s the diplomatic clause?”*
5. Click **Reset Knowledge Base** to clear the indexed embeddings and start fresh

#### Create Repair Ticket
- Enter ticket **title + description** → **Submit**
- Ticket is stored in **Neon/Postgres**

#### Create Rent Reminder
- Select a **due day (1–31)** + optional note → **Save**
- Reminder is stored in **Neon/Postgres**

#### Diagnostics
- **Test Neon connection** → checks database connectivity
- **API key detected** → shows if OpenAI key is available for the session

#### Clear Chat
- Clears the chat history for the current session (General Chat or Contract Chat)
---

## 📄 License

MIT License — free to use & modify.


