#!/usr/bin/env python3
"""
Tenant Chatbot Assistant (Single-file, single UI; lazy-load storage & RAG)
Run:
  streamlit run app_tenantbot_neon_single.py
"""

import os
import re
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

# ================== GLOBAL LIGHTWEIGHT STATE ==================
st.set_page_config(page_title="Tenant Chatbot", page_icon="🤖", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "page" not in st.session_state:
    st.session_state.page = "offline"
if "tickets" not in st.session_state:
    st.session_state.tickets = []
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "online_msgs" not in st.session_state:
    st.session_state.online_msgs = []
if "offline_msgs" not in st.session_state:
    st.session_state.offline_msgs = []
# 标记是否已手动初始化过数据库（仅改善 UX，不影响功能）
if "db_inited" not in st.session_state:
    st.session_state.db_inited = False
    
def apply_chat_input_visibility():
    """General Chat 与 Contract Chat 都显示 chat_input；其它页面隐藏。"""
    page = st.session_state.get("page", "offline")
    show = (page == "offline") or (page == "chat")
    st.markdown(
        f"""
        <style>
          div[data-testid='stChatInput'] {{
            display: {'block' if show else 'none'} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============== LAZY IMPORT HELPERS (关键) ===================
def lazy_import_psycopg():
    # 只有用到 DB 时才导入
    global psycopg2, psycopg2_extras
    try:
        import psycopg2  # type: ignore
        import psycopg2.extras as psycopg2_extras  # type: ignore
        return psycopg2, psycopg2_extras
    except Exception as e:
        raise RuntimeError(f"psycopg2 not available: {e}")

def lazy_import_langchain():
    # 只有进入合同问答页且需要创建索引/模型时才导入
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import FAISS
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory
        return {
            "PyPDFLoader": PyPDFLoader,
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
            "OpenAIEmbeddings": OpenAIEmbeddings,
            "ChatOpenAI": ChatOpenAI,
            "FAISS": FAISS,
            "ConversationalRetrievalChain": ConversationalRetrievalChain,
            "ConversationBufferMemory": ConversationBufferMemory,
        }
    except Exception as e:
        raise RuntimeError(
            "LangChain stack missing. Install:\n"
            "pip install langchain langchain-openai openai pypdf faiss-cpu\n"
            f"Details: {e}"
        )

# ================== DB HELPERS（惰性 + 短连接） =================
def get_db_conn():
    psycopg2, psycopg2_extras = lazy_import_psycopg()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        host = os.getenv("PG_HOST")
        db   = os.getenv("PG_DB")
        user = os.getenv("PG_USER")
        pwd  = os.getenv("PG_PASSWORD")
        port = os.getenv("PG_PORT", "5432")
        if not all([host, db, user, pwd]):
            raise RuntimeError("DATABASE_URL or PG_* env vars are not set.")
        dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{db}?sslmode=require"
    if "sslmode=" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(
        dsn,
        sslmode="require",
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        cursor_factory=lazy_import_psycopg()[1].DictCursor,  # psycopg2_extras.DictCursor
    )
    
# --- add: ensure tables exist, without any UI ---
def ensure_schema(conn):
    with conn.cursor() as cur:
        # repair_tickets
        cur.execute("SELECT to_regclass('public.repair_tickets');")
        exists = cur.fetchone()[0] is not None
        if not exists:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS repair_tickets (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

        # rent_reminders
        cur.execute("SELECT to_regclass('public.rent_reminders');")
        exists = cur.fetchone()[0] is not None
        if not exists:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rent_reminders (
                    id SERIAL PRIMARY KEY,
                    day_of_month INT NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
                    note TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

def init_db():
    psycopg2, _ = lazy_import_psycopg()
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS repair_tickets (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rent_reminders (
                    id SERIAL PRIMARY KEY,
                    day_of_month INT NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
                    note TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
    st.session_state.db_inited = True
    return True

def create_ticket(title: str, desc: str):
    with get_db_conn() as conn:
        ensure_schema(conn)                     # <— add
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO repair_tickets (title, description, status) VALUES (%s, %s, %s) RETURNING id;",
                (title, desc, "open"),
            )
            return cur.fetchone()["id"]

def list_tickets(limit: int = 50):
    with get_db_conn() as conn:
        ensure_schema(conn)                     # <— add
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, status, description, created_at
                FROM repair_tickets
                ORDER BY id DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()

def clear_tickets():
    with get_db_conn() as conn:
        ensure_schema(conn)                     # <— add
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE repair_tickets RESTART IDENTITY;")

def create_reminder(day_of_month: int, note: str):
    with get_db_conn() as conn:
        ensure_schema(conn)                     # <— add
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rent_reminders (day_of_month, note) VALUES (%s, %s) RETURNING id;",
                (day_of_month, note),
            )
            return cur.fetchone()["id"]

def list_reminders(limit: int = 20):
    with get_db_conn() as conn:
        ensure_schema(conn)                     # <— add
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, day_of_month, note, created_at
                FROM rent_reminders
                ORDER BY id DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()

def clear_reminders():
    with get_db_conn() as conn:
        ensure_schema(conn)                     # <— add
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE rent_reminders RESTART IDENTITY;")


# ================== RAG HELPERS（惰性导入） ====================
def build_vectorstore(uploaded_files):
    lc = lazy_import_langchain()
    paths = []
    try:
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getvalue())
                paths.append(tmp.name)
        docs = []
        for p in paths:
            loader = lc["PyPDFLoader"](p)
            docs += loader.load()
        splitter = lc["RecursiveCharacterTextSplitter"](chunk_size=900, chunk_overlap=180)
        texts = splitter.split_documents(docs)
        embeddings = lc["OpenAIEmbeddings"]()  # reads OPENAI_API_KEY
        vs = lc["FAISS"].from_documents(texts, embeddings)
        return vs
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except Exception:
                pass

def create_chain(vs):
    lc = lazy_import_langchain()
    last_err = None
    for m in ["gpt-4o-mini", "gpt-4o"]:
        try:
            llm = lc["ChatOpenAI"](model=m, temperature=0.2)
            memory = lc["ConversationBufferMemory"](
                memory_key="chat_history", return_messages=True, input_key="question", output_key="answer"
            )
            chain = lc["ConversationalRetrievalChain"].from_llm(
                llm=llm, retriever=vs.as_retriever(), memory=memory, return_source_documents=False
            )
            try: st.toast(f"Model in use: {m}")
            except Exception: pass
            return chain
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All models failed to init. Last error: {last_err}")

# ================== UTILITIES & SMALL-TALK ====================
def now_ts():
    return datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S")

def normalize_word(word: str) -> str:
    word = word.lower()
    suffixes = ["ing","ed","es","s","ly","tion","ions","ness","ment","ments","ities","ity","als","al","ers","er"]
    for suf in suffixes:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)]
    return word

def normalize_text(text: str) -> str:
    words = re.findall(r"[a-zA-Z\u4e00-\u9fff']+", text.lower())
    return " ".join(normalize_word(w) for w in words)

def normalize_text_zh(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\u4e00-\u9fff'，。！？、：；（）()《》“”\"' ]+", text))

def any_terms_en(text_norm: str, terms: list[str]) -> bool:
    for t in terms:
        t2 = normalize_word(t)
        if re.search(rf"\b{re.escape(t2)}\b", text_norm) or t2 in text_norm:
            return True
    return False

def contains_any_zh(text_norm: str, phrases: list[str]) -> bool:
    return any(p in text_norm for p in phrases)

def any_phrases(text: str, phrases: list[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(p) in norm for p in phrases)

def small_talk_zh(q_raw: str) -> str | None:
    q = normalize_text_zh(q_raw.strip())
    if contains_any_zh(q, ["你好","您好","嗨","哈喽","早上好","下午好","晚上好"]):
        return "你好！我是你的租客小助手 👋 有什么可以帮你的？"
    if contains_any_zh(q, ["你好吗","最近怎么样","最近如何","最近还好么"]):
        return "我很好，随时待命～你有什么想了解的？"
    if contains_any_zh(q, ["你是谁","你是干什么的","你叫什么名字"]):
        return "我是帮助租客进行简单咨询的聊天助手（离线模式）。"
    if contains_any_zh(q, ["谢谢","多谢","非常感谢","感谢你","太感谢了"]):
        return "不客气～还有什么我能帮忙的吗？"
    if contains_any_zh(q, ["能做什么","会干嘛","你能帮我什么","可以做什么"]):
        return "我可以进行问候与基础问答，并指引你创建报修或设置租金提醒。此离线版不支持合同问答。"
    if contains_any_zh(q, ["怎么开始","如何使用","怎么用","使用说明"]):
        return "你可以在侧栏切换语言或清空聊天记录。也可以问我打招呼、功能说明等基础问题。"
    if contains_any_zh(q, ["租金提醒","房租提醒","什么时候交房租","交租提醒"]):
        return "你可以自己每月记个备忘；完整版本里我可以替你保存提醒。"
    if contains_any_zh(q, ["报修","维修","漏水","坏了","修理","故障"]):
        return "请简单描述问题。完整版本中我可以帮你提交报修给物业。"
    return None

def small_talk_zh_basic(q_raw: str) -> str | None:
    q = normalize_text_zh(q_raw.strip())
    contract_like = ["合同","租约","条款","租金","押金","房东","租客","维修","报修","终止","违约","续约","账单","费用"]
    if contains_any_zh(q, contract_like):
        return None
    return small_talk_zh(q_raw)

def small_talk_response(q_raw: str) -> str | None:
    q = normalize_text(q_raw.strip())
    if any_terms_en(q, ["hi","hello","hey","morning","evening","afternoon"]) or any_phrases(q, ["你好","嗨","哈喽"]):
        return "Hello! I’m your Tenant Assistant 👋 How can I help you today?"
    if any_phrases(q, ["how are you","how's it going","how are u","how are ya","how are things","how do you feel","你好吗","最近怎么样","最近如何"]):
        return "I'm doing well and ready to help! How can I assist you today?"
    if any_phrases(q, ["who are you","what are you","your name","你是谁","你是干什么的"]):
        return "I’m a friendly chatbot that helps tenants understand contracts and manage repairs or rent reminders."
    if any_terms_en(q, ["thanks","thank","thx","appreciate"]) or any_phrases(q, ["thank you","many thanks","谢谢","多谢","非常感谢","感謝"]):
        return "You're welcome! If there’s anything else you need, just let me know."
    if any_phrases(q, ["what can you do","what can u do","能做什么","你会干嘛"]) or any_terms_en(q, ["function","feature","capability"]):
        return ("I can help you read tenancy agreements, create repair tickets, and set rent reminders. "
                "Once you add an API key, I can also answer contract questions directly!")
    if any_phrases(q, ["how to upload","upload pdf","add document","how to start","start upload","怎么上传","如何开始"]):
        return ("Click **‘Upload PDF contracts or house rules’** to add documents. "
                "Then click **‘Build/Refresh Knowledge Base’** after setting your API key.")
    if any_phrases(q, ["rent reminder","rent day","when to pay rent","租金提醒","什么时候交房租"]):
        return "Use **💰 Create Rent Reminder** in the sidebar to set a monthly reminder."
    if any_terms_en(q, ["repair","maintain","fix","broken","leak","damage","fault","issue"]) or any_phrases(q, ["报修","维修","漏水","坏了"]):
        return "Use **🧰 Create Repair Ticket** in the sidebar. Describe the problem and I’ll record it."
    if any_terms_en(q, ["contract","agreement","lease","term","clause","deposit","renewal","policy","rules"]) or any_phrases(q, ["合同","条款","押金","续约","租约"]):
        return "Upload your contract and set an API key; I’ll then answer questions based on the document."
    return None

def small_talk_response_basic(q_raw: str) -> str | None:
    q = normalize_text(q_raw.strip())
    if any_terms_en(q, ["contract","agreement","lease","tenant","landlord","deposit","repair","maintenance","damage","clause","policy","rent","renewal","notice","terminate"]):
        return None
    if any_terms_en(q, ["hi","hello","hey","morning","evening","afternoon"]) or any_phrases(q, ["你好","嗨","哈喽"]):
        return "Hello! I’m your Tenant Assistant 👋 How can I help you today?"
    if any_phrases(q, ["how are you","how's it going","how are u","how are ya","how are things","how do you feel","你好吗","最近怎么样","最近如何"]):
        return "I'm doing well and ready to help! How can I assist you today?"
    if any_terms_en(q, ["thanks","thank","thx","appreciate"]) or any_phrases(q, ["thank you","many thanks","谢谢","多谢","非常感谢","感謝"]):
        return "You're welcome! If there’s anything else you need, just let me know."
    if any_phrases(q, ["who are you","what are you","your name","你是谁","你是干什么的"]):
        return "I’m a friendly chatbot that helps tenants understand contracts and manage repairs or rent reminders."
    if any_phrases(q, ["what can you do","what can u do","能做什么","你会干嘛"]) or any_terms_en(q, ["function","feature","capability"]):
        return ("I can help you read tenancy agreements, create repair tickets, and set rent reminders. "
                "Once you add an API key, I can also answer contract questions directly!")
    if any_phrases(q, ["how to upload","upload pdf","add document","how to start","start upload","怎么上传","如何开始"]):
        return ("Click **‘Upload PDF contracts or house rules’** to add documents. "
                "Then click **‘Build/Refresh Knowledge Base’** after setting your API key.")
    return None

# ======================= SIDEBAR (单页面导航) ====================
st.markdown(
    """
<style>
.sidebar-btn {width:100%; text-align:left; background:#e3f2fd; border:none; padding:0.5rem 1rem; border-radius:0.5rem; margin:0.2rem 0;}
.sidebar-btn:hover {background:#bbdefb;}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🌐 Language / 语言")
    lang_choice = st.radio(
        "Select language / 选择语言",
        options=["English", "中文"],
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
    )
    st.session_state.lang = "en" if lang_choice == "English" else "zh"

    if st.session_state.lang == "en":
        btn_general = "💬 General Chat"
        btn_contract = "💬 Contract Chat"
        btn_ticket = "🧰 Create Repair Ticket"
        btn_reminder = "💰 Create Rent Reminder"
        caption_text = "Upload PDFs anytime. Build the knowledge base after setting OPENAI_API_KEY below."
        api_expander_label = "API Setup (for Contract Chat)"
        api_hint = "API key set for this session."
        clear_label = "🧹 Clear Chat"
        clear_success = "All chat history cleared."
    else:
        btn_general = "💬 普通聊天"
        btn_contract = "💬 合同问答"
        btn_ticket = "🧰 报修创建"
        btn_reminder = "💰 房租提醒"
        caption_text = "可随时上传 PDF。先在下方设置 OPENAI_API_KEY 再构建知识库。"
        api_expander_label = "API 设置（用于合同问答）"
        api_hint = "API 密钥已设置。"
        clear_label = "🧹 清空聊天"
        clear_success = "所有聊天记录已清空。"

    # 导航按钮
    if st.button(btn_general, use_container_width=True): st.session_state.page = "offline"
    if st.button(btn_contract, use_container_width=True): st.session_state.page = "chat"
    if st.button(btn_ticket, use_container_width=True): st.session_state.page = "ticket"
    if st.button(btn_reminder, use_container_width=True): st.session_state.page = "reminder"

    # --- API Setup ---
    with st.expander(api_expander_label):
        api_key_in = st.text_input("OpenAI API Key", type="password")
        if api_key_in:
            os.environ["OPENAI_API_KEY"] = api_key_in
            st.success(api_hint)

    # # ✅ 把 Clear Chat 放到 API Setup 下面（Diagnostics 之前）
    # if st.button(clear_label, use_container_width=True):
    #     st.session_state.offline_msgs.clear()
    #     st.session_state.online_msgs.clear()
    #     st.session_state.pop("vectorstore", None)
    #     st.session_state.pop("chain", None)
    #     chain = st.session_state.get("chain")
    #     if chain and getattr(chain, "memory", None):
    #         try:
    #             chain.memory.clear()
    #         except Exception:
    #             pass
    #     st.success(clear_success)
    #     # st.rerun()

    # st.caption(caption_text)
    # st.divider()
    
    # ✅ Clear Chat：只清对话，不清知识库
    if st.button(clear_label, use_container_width=True, key="btn_clear_chat"):
        # 清空对话历史
        st.session_state.offline_msgs = []
        st.session_state.online_msgs = []
        # 不要动向量库/链，避免合同页输入框被禁用
        st.success(clear_success)
    
    # （可选）单独的“重置知识库”按钮
    # reset_kb_label = "♻️ Reset Knowledge Base" if st.session_state.lang != "zh" else "♻️ 重置知识库"
    # if st.button(reset_kb_label, use_container_width=True, key="btn_reset_kb"):
    #     st.session_state.pop("vectorstore", None)
    #     st.session_state.pop("chain", None)
    #     # 若用了链的 memory，可安全清一次
    #     chain = st.session_state.get("chain")
    #     if chain and getattr(chain, "memory", None):
    #         try:
    #             chain.memory.clear()
    #         except Exception:
    #             pass
    #     st.success("Knowledge base reset. Build it again to ask questions." 
    #                if st.session_state.lang != "zh" 
    #                else "知识库已重置，请重新构建后再提问。")

    st.caption(caption_text)
    st.divider()

    # # --- Diagnostics（留在最后）---
    # with st.expander("🧪 Diagnostics (on-demand)"):
    #     if st.button("Test Neon connection"):
    #         try:
    #             with get_db_conn() as conn:
    #                 with conn.cursor() as cur:
    #                     cur.execute("SELECT NOW();")
    #             st.success("DB connected ✔️")
    #         except Exception as e:
    #             st.error(f"DB connect failed: {e}")
    #     st.write("API Key detected:", bool(os.getenv("OPENAI_API_KEY")))
    
    # --- Diagnostics（留在最后）---
    if st.session_state.lang == "zh":
        diag_label = "🧪 诊断（按需执行）"
        test_label = "测试 Neon 数据库连接"
        db_success = "数据库连接成功 ✔️"
        db_fail = "数据库连接失败："
        api_label = "检测到 API Key："
    else:
        diag_label = "🧪 Diagnostics (on-demand)"
        test_label = "Test Neon connection"
        db_success = "DB connected ✔️"
        db_fail = "DB connect failed: "
        api_label = "API Key detected:"

    with st.expander(diag_label):
        if st.button(test_label):
            try:
                with get_db_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT NOW();")
                st.success(db_success)
            except Exception as e:
                st.error(f"{db_fail}{e}")

        st.write(f"{api_label} {bool(os.getenv('OPENAI_API_KEY'))}")
        
# —— Sidebar 结束后立刻调用，确保每次切页先隐藏/显示 chat 输入条
apply_chat_input_visibility()

# ========================= PAGES（单文件内切换） =========================

# # --- Contract Chat ---
# if st.session_state.page == "chat":
#     is_zh = st.session_state.lang == "zh"
#     st.title("租客聊天助手" if is_zh else "Tenant Chatbot Assistant")
#     st.caption("基于已上传的租赁合同进行问答" if is_zh else "Contract-aware Q&A using uploaded tenancy documents.")

#     uploaded = st.file_uploader(
#         "上传租赁合同或房屋守则（PDF）" if is_zh else "Upload PDF contracts or house rules",
#         type="pdf", accept_multiple_files=True
#     )
#     if uploaded:
#         build_disabled = not bool(os.getenv("OPENAI_API_KEY"))
#         clicked = st.button(
#             "🔄 构建/刷新知识库" if is_zh else "🔄 Build/Refresh Knowledge Base",
#             disabled=build_disabled,
#             help=("请先设置 OPENAI_API_KEY" if build_disabled else "根据 PDF 构建 FAISS 索引") if is_zh else
#                  ("Set OPENAI_API_KEY first" if build_disabled else "Build FAISS index from PDFs"),
#         )
#         if clicked:
#             with st.spinner("正在根据文档构建索引…" if is_zh else "Indexing documents…"):
#                 vs = build_vectorstore(uploaded)
#                 st.session_state.vectorstore = vs
#                 st.session_state.chain = create_chain(vs)
#             st.success("知识库已就绪！现在可以在下方提问。" if is_zh else "Knowledge base ready! Ask questions below.")

#     if "chain" in st.session_state:
#         for m in st.session_state.online_msgs:
#             with st.chat_message(m["role"]):
#                 if m.get("ts"):
#                     st.caption(m["ts"])
#                 st.markdown(m["content"])
#         user_q = st.chat_input("就你的合同提问…" if is_zh else "Ask about your contract…")
#         if user_q:
#             ts_user = now_ts()
#             st.session_state.online_msgs.append({"role": "user", "content": user_q, "ts": ts_user})
#             with st.chat_message("user"):
#                 st.caption(ts_user)
#                 st.markdown(user_q)

#             smalltalk = small_talk_zh_basic(user_q) if is_zh else small_talk_response_basic(user_q)
#             if smalltalk is not None:
#                 final_md = smalltalk
#             else:
#                 with st.spinner("正在回答…" if is_zh else "Answering…"):
#                     try:
#                         system_hint = (
#                             "你是一名租客助手。仅根据已上传文档作答；若文档中没有答案，请说明信息不足。"
#                             if is_zh else
#                             "You are a helpful Tenant Assistant. Answer ONLY based on the uploaded documents. "
#                             "If the answer isn't present in the documents, say you don't have enough information."
#                         )
#                         query = f"{system_hint}\nQuestion: {user_q}"
#                         resp = st.session_state.chain.invoke({"question": query})
#                         final_md = resp.get("answer", "（暂无答案）" if is_zh else "(no answer)")
#                     except Exception as e:
#                         msg = str(e)
#                         if "insufficient_quota" in msg or "429" in msg:
#                             final_md = "（模型额度不足或达到速率限制）" if is_zh else "Quota/rate limit hit."
#                         elif "401" in msg or "invalid_api_key" in msg.lower():
#                             final_md = "（API Key 无效）" if is_zh else "Invalid API key."
#                         else:
#                             final_md = f"（RAG 调用失败：{e}）" if is_zh else f"RAG call failed: {e}"

#             ts_ans = now_ts()
#             st.session_state.online_msgs.append({"role": "assistant", "content": final_md, "ts": ts_ans})
#             with st.chat_message("assistant"):
#                 st.caption(ts_ans)
#                 st.markdown(final_md)
#     else:
#         st.info(
#             "设置 API Key 并构建知识库后开始提问。" if is_zh
#             else "Set your API key and build the knowledge base to start asking questions."
#         )

# --- Contract Chat ---
if st.session_state.page == "chat":
    is_zh = st.session_state.lang == "zh"
    st.title("租客聊天助手" if is_zh else "Tenant Chatbot Assistant")
    st.caption("基于已上传的租赁合同进行问答" if is_zh else "Contract-aware Q&A using uploaded tenancy documents.")

    uploaded = st.file_uploader(
        "上传租赁合同或房屋守则（PDF）" if is_zh else "Upload PDF contracts or house rules",
        type="pdf", accept_multiple_files=True
    )
    # if uploaded:
    #     build_disabled = not bool(os.getenv("OPENAI_API_KEY"))
    #     clicked = st.button(
    #         "🔄 构建/刷新知识库" if is_zh else "🔄 Build/Refresh Knowledge Base",
    #         disabled=build_disabled,
    #         help=( "请先设置 OPENAI_API_KEY" if build_disabled else "根据 PDF 构建 FAISS 索引" )
    #              if is_zh else ( "Set OPENAI_API_KEY first" if build_disabled else "Build FAISS index from PDFs" ),
    #     )
    #     if clicked:
    #         with st.spinner("正在根据文档构建索引…" if is_zh else "Indexing documents…"):
    #             vs = build_vectorstore(uploaded)
    #             st.session_state.vectorstore = vs
    #             st.session_state.chain = create_chain(vs)
    #         st.success("知识库已就绪！现在可以在下方提问。" if is_zh else "Knowledge base ready! Ask questions below.")
        
    if uploaded:
        build_disabled = not bool(os.getenv("OPENAI_API_KEY"))

        col1, col2 = st.columns([3, 2])
        with col1:
            clicked = st.button(
                "🔄 构建/刷新知识库" if is_zh else "🔄 Build/Refresh Knowledge Base",
                disabled=build_disabled,
                help=("请先设置 OPENAI_API_KEY" if build_disabled else "根据 PDF 构建 FAISS 索引")
                if is_zh
                else ("Set OPENAI_API_KEY first" if build_disabled else "Build FAISS index from PDFs"),
                use_container_width=True,
            )
        with col2:
            reset_clicked = st.button(
                "♻️ 重置知识库" if is_zh else "♻️ Reset Knowledge Base",
                use_container_width=True,
            )

        # ===== 构建知识库 =====
        if clicked:
            with st.spinner("正在根据文档构建索引…" if is_zh else "Indexing documents…"):
                vs = build_vectorstore(uploaded)
                st.session_state.vectorstore = vs
                st.session_state.chain = create_chain(vs)
            st.success("知识库已就绪！现在可以在下方提问。" if is_zh else "Knowledge base ready! Ask questions below.")

        # ===== 重置知识库 =====
        if reset_clicked:
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("chain", None)
            chain = st.session_state.get("chain")
            if chain and getattr(chain, "memory", None):
                try:
                    chain.memory.clear()
                except Exception:
                    pass
            st.info("知识库已重置，请重新构建后再提问。" if is_zh else "Knowledge base has been reset. Please rebuild before asking questions.")

    # —— 关键改动：无论是否已建库，都先渲染历史，然后渲染一个输入框
    has_chain = st.session_state.get("chain") is not None

    # 历史消息（有就显示）
    for m in st.session_state.get("online_msgs", []):
        with st.chat_message(m["role"]):
            if m.get("ts"):
                st.caption(m["ts"])
            st.markdown(m["content"])

    # 输入框：未建库时禁用并提示；建库后可用
    ph_ready = "就你的合同提问…" if is_zh else "Ask about your contract…"
    ph_build = "请先构建知识库…" if is_zh else "Build the knowledge base first…"
    user_q = st.chat_input(ph_ready if has_chain else ph_build, disabled=not has_chain)

    if has_chain and user_q:
        ts_user = now_ts()
        st.session_state.online_msgs.append({"role": "user", "content": user_q, "ts": ts_user})
        with st.chat_message("user"):
            st.caption(ts_user)
            st.markdown(user_q)

        smalltalk = small_talk_zh_basic(user_q) if is_zh else small_talk_response_basic(user_q)
        if smalltalk is not None:
            final_md = smalltalk
        else:
            with st.spinner("正在回答…" if is_zh else "Answering…"):
                try:
                    system_hint = (
                        "你是一名租客助手。仅根据已上传文档作答；若文档中没有答案，请说明信息不足。"
                        if is_zh else
                        "You are a helpful Tenant Assistant. Answer ONLY based on the uploaded documents. "
                        "If the answer isn't present in the documents, say you don't have enough information."
                    )
                    query = f"{system_hint}\nQuestion: {user_q}"
                    resp = st.session_state.chain.invoke({"question": query})
                    final_md = resp.get("answer", "（暂无答案）" if is_zh else "(no answer)")
                except Exception as e:
                    msg = str(e)
                    if "insufficient_quota" in msg or "429" in msg:
                        final_md = "（模型额度不足或达到速率限制）" if is_zh else "Quota/rate limit hit."
                    elif "401" in msg or "invalid_api_key" in msg.lower():
                        final_md = "（API Key 无效）" if is_zh else "Invalid API key."
                    else:
                        final_md = f"（RAG 调用失败：{e}）" if is_zh else f"RAG call failed: {e}"

        ts_ans = now_ts()
        st.session_state.online_msgs.append({"role": "assistant", "content": final_md, "ts": ts_ans})
        with st.chat_message("assistant"):
            st.caption(ts_ans)
            st.markdown(final_md)

    if not has_chain:
        st.info(
            "设置 API Key 并构建知识库后开始提问。" if is_zh
            else "Set your API key and build the knowledge base to start asking questions."
        )

# --- Repair Ticket ---
elif st.session_state.page == "ticket":
    is_zh = st.session_state.lang == "zh"
    st.title("🧰 创建报修工单" if is_zh else "🧰 Create Repair Ticket")

    with st.form("ticket_form", clear_on_submit=True):
        t_title = st.text_input("问题标题" if is_zh else "Issue title",
                                placeholder="厨房水槽漏水" if is_zh else "Leaking sink in kitchen")
        t_desc  = st.text_area("问题描述" if is_zh else "Description",
                               placeholder="请描述具体情况…" if is_zh else "Describe the issue…")
        submitted = st.form_submit_button("📨 提交报修" if is_zh else "📨 Submit Ticket")
        if submitted:
            if not t_title.strip():
                st.warning("请填写问题标题。" if is_zh else "Please enter a title.")
            else:
                try:
                    new_id = create_ticket(t_title.strip(), t_desc.strip())
                    st.success(("报修已保存到数据库！" if is_zh else "Ticket saved to database!") + f"  (#{new_id})")
                except Exception as e:
                    st.error(f"DB error: {e}")

    st.subheader("我的报修工单" if is_zh else "My Tickets")
    if st.button("🗑️ 清除所有报修记录" if is_zh else "🗑️ Clear All Tickets"):
        try:
            clear_tickets(); st.success("已删除！" if is_zh else "All tickets deleted!"); st.rerun()
        except Exception as e:
            st.error(f"DB delete error: {e}")

    try:
        rows = list_tickets()
    except Exception as e:
        rows = []
        st.error(f"DB read error: {e}")

    if not rows:
        st.caption("暂无工单" if is_zh else "No tickets yet")
    else:
        tz = ZoneInfo("Asia/Singapore")
        for r in rows:
            created_local = r["created_at"].astimezone(tz)
            ts_str = created_local.strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**#{r['id']} – {r['title']}** — _{r['status']}_")
            if r["description"]:
                st.caption(r["description"])
            st.caption(f"Created at: {ts_str} (SGT)")

# --- Rent Reminder ---
elif st.session_state.page == "reminder":
    is_zh = st.session_state.lang == "zh"
    st.title("💰 创建房租提醒" if is_zh else "💰 Create Rent Reminder")

    with st.form("reminder_form", clear_on_submit=True):
        r_day  = st.number_input("每月几号" if is_zh else "Due day of month", 1, 31, 1)
        r_note = st.text_input("备注" if is_zh else "Note",
                               placeholder="通过银行卡尾号••1234转账" if is_zh else "Pay via bank transfer ending ••1234")
        r_submit = st.form_submit_button("💾 保存提醒" if is_zh else "💾 Save Reminder")
        if r_submit:
            try:
                rid = create_reminder(int(r_day), (r_note or "").strip())
                st.success(("提醒已保存到数据库！" if is_zh else "Reminder saved to database!") + f"  (#{rid})")
            except Exception as e:
                st.error(f"DB error: {e}")

    st.subheader("当前提醒" if is_zh else "Current Reminder")
    if st.button("🗑️ 清除所有提醒" if is_zh else "🗑️ Clear All Reminders"):
        try:
            clear_reminders(); st.success("已清空！" if is_zh else "All reminders deleted!"); st.rerun()
        except Exception as e:
            st.error(f"DB delete error: {e}")

    try:
        rows = list_reminders()
    except Exception as e:
        rows = []
        st.error(f"DB read error: {e}")

    if not rows:
        st.caption("暂无提醒" if is_zh else "No reminders yet")
    else:
        tz = ZoneInfo("Asia/Singapore")
        for r in rows:
            created_local = r["created_at"].astimezone(tz)
            ts_str = created_local.strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"每月的第 **{r['day_of_month']}** 天 — {r['note'] or '—'}" if is_zh
                     else f"Every month on day **{r['day_of_month']}** — {r['note'] or '—'}")
            st.caption(f"Created at: {ts_str} (SGT)")

# --- General Chat (Offline) ---
elif st.session_state.page == "offline":
    is_zh = st.session_state.lang == "zh"
    st.title("💬 通用离线聊天" if is_zh else "💬 General Chat (Offline)")
    st.caption("无需 API，仅支持基础闲聊与引导。" if is_zh else "No API required. Small talk and quick help only.")

    for m in st.session_state.offline_msgs:
        with st.chat_message(m["role"]):
            if m.get("ts"):
                st.caption(m["ts"])
            st.markdown(m["content"])

    user_q = st.chat_input("打个招呼或问一些基础问题…" if is_zh else "Say hello or ask about some basic information…")
    if user_q:
        ts_now = now_ts()
        st.session_state.offline_msgs.append({"role": "user", "content": user_q, "ts": ts_now})
        with st.chat_message("user"):
            st.caption(ts_now)
            st.markdown(user_q)
        ans = (small_talk_zh(user_q) if is_zh else small_talk_response(user_q)) or (
            "当前为离线聊天模式。你也可以在侧栏切换到“合同问答”。" if is_zh else
            "I'm in offline chat mode. Use the sidebar to switch features."
        )
        ts_ans = now_ts()
        st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
        with st.chat_message("assistant"):
            st.caption(ts_ans)
            st.markdown(ans)
