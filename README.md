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

streamlit run app_tenantbot_neon_single.py

