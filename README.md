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

| 功能 | Feature Description |
|------|----------------------|
| 💬 General Chat (offline) | Fast responses, no API needed |
| 📄 Contract Q&A | Upload tenancy agreement PDF → Build FAISS → Ask questions |
| 🧰 Repair Tickets | Tenant submits repair requests; stored in DB |
| 💰 Rent Reminders | Save rent reminder (Day-of-month + note) |
| 🌐 Multi-language UI | English / 中文 toggle from sidebar |
| 🔧 Diagnostics | Test Neon DB connection + detect OpenAI API |

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
## 🧠 RAG Pipeline (Contract Q&A)
PDF Upload → PyPDFLoader → Text Splitter → OpenAI Embeddings → FAISS Vectorstore
                     ↓
          ConversationalRetrievalChain
                     ↓
               Contract Answer


