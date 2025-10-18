#!/usr/bin/env python3
"""
Tenant Chatbot Assistant (Streamlit + RAG + Neon/Postgres persistence)
Run:
  streamlit run app_tenantbot_neon.py
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

import os
import tempfile
import re
from datetime import datetime
import streamlit as st

# ---------- Database (Neon/Postgres via psycopg2) ----------
from contextlib import closing
import psycopg2
import psycopg2.extras


# ---------- Optional env loader ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- LangChain / RAG ----------
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False



# ---------- Database (Neon/Postgres via psycopg2) ----------

def get_db_conn():
    """
    每次返回一个**新的**连接（不要缓存），强制 sslmode=require，并打开 keepalive。
    建议 DATABASE_URL 使用 *-pooler 主机名。
    """
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
        cursor_factory=psycopg2.extras.DictCursor,
    )

def init_db():
    with closing(get_db_conn()) as conn:
        with conn, conn.cursor() as cur:
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
    return True


# ---------- CRUD helpers（短连接） ----------

def create_ticket(title: str, desc: str):
    with closing(get_db_conn()) as conn:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO repair_tickets (title, description, status) VALUES (%s, %s, %s) RETURNING id;",
                (title, desc, "open"),
            )
            return cur.fetchone()["id"]

def list_tickets(limit: int = 50):
    with closing(get_db_conn()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, status, description, created_at
                FROM repair_tickets
                ORDER BY id DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()

def create_reminder(day_of_month: int, note: str):
    with closing(get_db_conn()) as conn:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rent_reminders (day_of_month, note) VALUES (%s, %s) RETURNING id;",
                (day_of_month, note),
            )
            return cur.fetchone()["id"]

def list_reminders(limit: int = 20):
    with closing(get_db_conn()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, day_of_month, note, created_at
                FROM rent_reminders
                ORDER BY id DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()

# ---------- Page config ----------
st.set_page_config(
    page_title="Tenant Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
<style>
.sidebar-btn {width:100%; text-align:left; background:#e3f2fd; border:none; padding:0.5rem 1rem; border-radius:0.5rem; margin:0.2rem 0;}
.sidebar-btn:hover {background:#bbdefb;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Init DB (once) ----------
try:
    init_db()
except Exception as e:
    st.sidebar.error(f"DB init failed: {e}")
    
    
# 放在 import 后、函数定义后
if "db_inited" not in st.session_state:
    try:
        init_db()        # 里面是 with closing(get_db_conn())：用完即关
        st.session_state.db_inited = True
    except Exception as e:
        st.sidebar.error(f"DB init failed: {e}")

# ---------------------- Session Init ----------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"  # "en" or "zh"
if "page" not in st.session_state:
    st.session_state.page = "offline"  # default landing page
if "tickets" not in st.session_state:
    st.session_state.tickets = []
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "online_msgs" not in st.session_state:
    st.session_state.online_msgs = []

# --- sidebar navigation ---
with st.sidebar:
    # ====== Language Switch ======
    st.header("🌐 Language / 语言")
    lang_choice = st.radio(
        "Select language / 选择语言",
        options=["English", "中文"],
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
    )
    st.session_state.lang = "en" if lang_choice == "English" else "zh"

    # ====== Sidebar Titles by Language ======
    if st.session_state.lang == "en":
        st.header("🏠 Tenant Utilities")
        api_expander_label = "API Setup"
        api_hint = "API key set for this session. Now you can build the knowledge base."
        btn_general = "💬 General Chat"
        btn_contract = "💬 Contract Chat"
        btn_ticket = "🧰 Create Repair Ticket"
        btn_reminder = "💰 Create Rent Reminder"
        caption_text = (
            "Upload PDFs anytime. Enable the build button by setting OPENAI_API_KEY in the expander above or via .env."
        )
        clear_label = "🧹 Clear Chat"
        clear_success = "All chat history cleared."
    else:
        st.header("🏠 租客助手功能")
        api_expander_label = "API 设置"
        api_hint = "API 密钥已设置，可建立知识库。"
        btn_general = "💬 普通聊天"
        btn_contract = "💬 合同问答"
        btn_ticket = "🧰 报修创建"
        btn_reminder = "💰 房租提醒"
        caption_text = "可随时上传 PDF。设置 OPENAI_API_KEY 后启用知识库构建按钮。"
        clear_label = "🧹 清空聊天"
        clear_success = "所有聊天记录已清空。"

    # ====== API Setup ======
    with st.expander(api_expander_label):
        api_key_in = st.text_input("OpenAI API Key", type="password")
        if api_key_in:
            os.environ["OPENAI_API_KEY"] = api_key_in
            st.success(api_hint)

    # ====== Navigation Buttons ======
    if st.button(btn_general, key="offline_btn", use_container_width=True):
        st.session_state.page = "offline"
    if st.button(btn_contract, key="chat_btn", use_container_width=True):
        st.session_state.page = "chat"
    if st.button(btn_ticket, key="ticket_btn", use_container_width=True):
        st.session_state.page = "ticket"
    if st.button(btn_reminder, key="reminder_btn", use_container_width=True):
        st.session_state.page = "reminder"

    # ====== Footer ======
    st.caption(caption_text)
    st.divider()

    # ====== Diagnostics ======
    with st.expander("🧪 Diagnostics"):
        st.caption("Click to run checks. They are skipped by default to keep the app snappy.")
        run_diag = st.button("▶️ Run diagnostics")
        if run_diag:
            try:
                from contextlib import closing
                with closing(get_db_conn()) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT NOW();")
                st.success("DB connected ✔️")
            except Exception as e:
                st.error(f"DB connect failed: {e}")
        st.write("LangChain imports ok:", LANGCHAIN_AVAILABLE)
        st.write("API Key detected:", bool(os.getenv("OPENAI_API_KEY")))

    # ====== Clear Chat ======
    if st.button(clear_label, use_container_width=True):
        st.session_state.setdefault("offline_msgs", [])
        st.session_state.setdefault("online_msgs", [])
        st.session_state.offline_msgs.clear()
        st.session_state.online_msgs.clear()
        # also reset vectorstore/chain
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("chain", None)
        chain = st.session_state.get("chain")
        if chain and getattr(chain, "memory", None):
            try:
                chain.memory.clear()
            except Exception:
                pass
        st.success(clear_success)
        st.rerun()

# --- core chat functions ---
def build_vectorstore(uploaded_files):
    """Build FAISS vectorstore from uploaded PDF files."""
    paths = []
    try:
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getvalue())
                paths.append(tmp.name)
        docs = []
        for p in paths:
            loader = PyPDFLoader(p)
            docs += loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
        texts = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()  # reads OPENAI_API_KEY
        vs = FAISS.from_documents(texts, embeddings)
        return vs
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except Exception:
                pass


def create_chain(vs):
    # Model fallback chain (try gpt-4o-mini, then gpt-4o)
    last_err = None
    for m in ["gpt-4o-mini", "gpt-4o"]:
        try:
            llm = ChatOpenAI(model=m, temperature=0.2)
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="question",
                output_key="answer",
            )
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vs.as_retriever(),
                memory=memory,
                return_source_documents=False,
            )
            try:
                st.toast(f"Model in use: {m}")
            except Exception:
                pass
            return chain
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All models failed to init. Last error: {last_err}")


# ---------------------- Utilities ----------------------

def now_ts(lang: str) -> str:
    """Return local timestamp (SGT assumed) formatted to seconds."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===== Small-talk helpers (shared) =====
def normalize_word(word: str) -> str:
    word = word.lower()
    suffixes = [
        "ing",
        "ed",
        "es",
        "s",
        "ly",
        "tion",
        "ions",
        "ness",
        "ment",
        "ments",
        "ities",
        "ity",
        "als",
        "al",
        "ers",
        "er",
    ]
    for suf in suffixes:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)]
    return word


def normalize_text(text: str) -> str:
    words = re.findall(r"[a-zA-Z\u4e00-\u9fff']+", text.lower())
    return " ".join(normalize_word(w) for w in words)


def normalize_text_zh(text: str) -> str:
    return "".join(
        re.findall(r"[0-9A-Za-z\u4e00-\u9fff'，。！？、：；（）()《》“”\"' ]+", text)
    )


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

# ---------------------- Small-talk (ZH) ----------------------
def small_talk_zh(q_raw: str) -> str | None:
    q = normalize_text_zh(q_raw.strip())
    if contains_any_zh(q, ["你好", "您好", "嗨", "哈喽", "早上好", "下午好", "晚上好"]):
        return "你好！我是你的租客小助手 👋 有什么可以帮你的？"
    if contains_any_zh(q, ["你好吗", "最近怎么样", "最近如何", "最近还好么"]):
        return "我很好，随时待命～你有什么想了解的？"
    if contains_any_zh(q, ["你是谁", "你是干什么的", "你叫什么名字"]):
        return "我是帮助租客进行简单咨询的聊天助手（离线模式）。"
    if contains_any_zh(q, ["谢谢", "多谢", "非常感谢", "感谢你", "太感谢了"]):
        return "不客气～还有什么我能帮忙的吗？"
    if contains_any_zh(q, ["能做什么", "会干嘛", "你能帮我什么", "可以做什么"]):
        return "我可以进行问候与基础问答，并指引你创建报修或设置租金提醒。此离线版不支持合同问答。"
    if contains_any_zh(q, ["怎么开始", "如何使用", "怎么用", "使用说明"]):
        return "你可以在侧栏切换语言进行或清空聊天记录。也可以问我打招呼、功能说明等基础问题。"
    if contains_any_zh(q, ["租金提醒", "房租提醒", "什么时候交房租", "交租提醒"]):
        return "你可以自己每月记个备忘；完整版本里我可以替你保存提醒。"
    if contains_any_zh(q, ["报修", "维修", "漏水", "坏了", "修理", "故障"]):
        return "请简单描述问题。完整版本中我可以帮你提交报修给物业。"
    return None


def small_talk_zh_basic(q_raw: str) -> str | None:
    q = normalize_text_zh(q_raw.strip())
    contract_like = ["合同", "租约", "条款", "租金", "押金", "房东", "租客", "维修", "报修", "终止", "违约", "续约", "账单", "费用"]
    if contains_any_zh(q, contract_like):
        return None
    return small_talk_zh(q_raw)


# ---------------------- Small-talk (EN) ----------------------
def small_talk_response(q_raw: str) -> str | None:
    q = normalize_text(q_raw.strip())
    if any_terms_en(q, ["hi", "hello", "hey", "morning", "evening", "afternoon"]) or any_phrases(q, ["你好", "嗨", "哈喽"]):
        return "Hello! I’m your Tenant Assistant 👋 How can I help you today?"
    if any_phrases(q, ["how are you", "how's it going", "how are u", "how are ya", "how are things", "how do you feel", "你好吗", "最近怎么样", "最近如何"]):
        return "I'm doing well and ready to help! How can I assist you today?"
    if any_phrases(q, ["who are you", "what are you", "your name", "你是谁", "你是干什么的"]):
        return "I’m a friendly chatbot that helps tenants understand contracts and manage repairs or rent reminders."
    if any_terms_en(q, ["thanks", "thank", "thx", "appreciate"]) or any_phrases(q, ["thank you", "many thanks", "谢谢", "多谢", "非常感谢", "感謝"]):
        return "You're welcome! If there’s anything else you need, just let me know."
    if any_phrases(q, ["what can you do", "what can u do", "能做什么", "你会干嘛"]) or any_terms_en(q, ["function", "feature", "capability"]):
        return (
            "I can help you read tenancy agreements, create repair tickets, and set rent reminders. "
            "Once you add an API key, I can also answer contract questions directly!"
        )
    if any_phrases(q, ["how to upload", "upload pdf", "add document", "how to start", "start upload", "怎么上传", "如何开始"]):
        return (
            "Click **‘Upload PDF contracts or house rules’** to add documents. "
            "Then click **‘Build/Refresh Knowledge Base’** after setting your API key."
        )
    if any_phrases(q, ["rent reminder", "rent day", "when to pay rent", "租金提醒", "什么时候交房租"]):
        return "Use **💰 Create Rent Reminder** in the sidebar to set a monthly reminder."
    if any_terms_en(q, ["repair", "maintain", "fix", "broken", "leak", "damage", "fault", "issue"]) or any_phrases(q, ["报修", "维修", "漏水", "坏了"]):
        return "Use **🧰 Create Repair Ticket** in the sidebar. Describe the problem and I’ll record it."
    if any_terms_en(q, ["contract", "agreement", "lease", "term", "clause", "deposit", "renewal", "policy", "rules"]) or any_phrases(q, ["合同", "条款", "押金", "续约", "租约"]):
        return "Upload your contract and set an API key; I’ll then answer questions based on the document."
    return None


def small_talk_response_basic(q_raw: str) -> str | None:
    q = normalize_text(q_raw.strip())
    if any_terms_en(
        q,
        [
            "contract",
            "agreement",
            "lease",
            "tenant",
            "landlord",
            "deposit",
            "repair",
            "maintenance",
            "damage",
            "clause",
            "policy",
            "rent",
            "renewal",
            "notice",
            "terminate",
        ],
    ):
        return None
    if any_terms_en(q, ["hi", "hello", "hey", "morning", "evening", "afternoon"]) or any_phrases(q, ["你好", "嗨", "哈喽"]):
        return "Hello! I’m your Tenant Assistant 👋 How can I help you today?"
    if any_phrases(q, ["how are you", "how's it going", "how are u", "how are ya", "how are things", "how do you feel", "你好吗", "最近怎么样", "最近如何"]):
        return "I'm doing well and ready to help! How can I assist you today?"
    if any_terms_en(q, ["thanks", "thank", "thx", "appreciate"]) or any_phrases(q, ["thank you", "many thanks", "谢谢", "多谢", "非常感谢", "感謝"]):
        return "You're welcome! If there’s anything else you need, just let me know."
    if any_phrases(q, ["who are you", "what are you", "your name", "你是谁", "你是干什么的"]):
        return "I’m a friendly chatbot that helps tenants understand contracts and manage repairs or rent reminders."
    if any_phrases(q, ["what can you do", "what can u do", "能做什么", "你会干嘛"]) or any_terms_en(q, ["function", "feature", "capability"]):
        return (
            "I can help you read tenancy agreements, create repair tickets, and set rent reminders. "
            "Once you add an API key, I can also answer contract questions directly!"
        )
    if any_phrases(q, ["how to upload", "upload pdf", "add document", "how to start", "start upload", "怎么上传", "如何开始"]):
        return (
            "Click **‘Upload PDF contracts or house rules’** to add documents. "
            "Then click **‘Build/Refresh Knowledge Base’** after setting your API key."
        )
    return None

# ===================== PAGES =====================
# --- page: contract chat ---
if st.session_state.page == "chat":
    lang = st.session_state.get("lang", "en")

    if lang == "zh":
        title = "租客聊天助手"
        subtitle = "基于已上传的租赁合同进行问答"
        upload_label = "上传租赁合同或房屋守则（PDF）"
        build_btn = "🔄 构建/刷新知识库"
        build_help_on = "根据 PDF 构建 FAISS 索引"
        build_help_off = "请先在侧栏的『API 设置』中填写 OPENAI_API_KEY 才能构建索引"
        offline_banner = "💬 离线聊天模式：在设置 API Key 之前，你仍然可以和我打个招呼。"
        api_hint = "检测到 API Key。请上传 PDF 并点击『构建/刷新知识库』以启用合同问答。你也可以先小聊一下。"
        chat_ph_offline = "打个招呼或问一些基础问题…"
        chat_ph_online = "就你的合同提问…"
        dep_missing = "未安装 LangChain。请运行：pip install langchain langchain-openai openai pypdf faiss-cpu"
        idx_spinner = "正在根据文档构建索引…"
        idx_done = "知识库已就绪！现在可以在下方提问。"
        ans_spinner = "正在回答…"
        offline_hint = "当前为离线聊天模式。你也可以在侧栏切换到『合同问答』。"
    else:
        title = "Tenant Chatbot Assistant"
        subtitle = "Contract-aware Q&A using uploaded tenancy documents."
        upload_label = "Upload PDF contracts or house rules"
        build_btn = "🔄 Build/Refresh Knowledge Base"
        build_help_on = "Build FAISS index from PDFs"
        build_help_off = "Set OPENAI_API_KEY in the sidebar to enable indexing"
        offline_banner = "💬 Offline Chat Mode: You can still say hi while waiting to set your API key."
        api_hint = (
            "API key detected. Upload PDFs and click **Build/Refresh Knowledge Base** to enable contract Q&A. "
            "You can still have a quick small talk below."
        )
        chat_ph_offline = "Say hello or ask about some basic information…"
        chat_ph_online = "Ask about your contract…"
        dep_missing = "LangChain not installed. Run: pip install langchain langchain-openai openai pypdf faiss-cpu"
        idx_spinner = "Indexing documents…"
        idx_done = "Knowledge base ready! Ask questions below."
        ans_spinner = "Answering…"
        offline_hint = (
            "I'm in offline chat mode. You can explore the sidebar features, "
            "or switch to Contract Chat for document-based Q&A."
        )

    st.title(f"🤖 {title}")
    st.caption(subtitle)

    if not LANGCHAIN_AVAILABLE:
        st.error(dep_missing)
        st.stop()

    uploaded = st.file_uploader(upload_label, type="pdf", accept_multiple_files=True)
    if uploaded:
        build_disabled = not bool(os.getenv("OPENAI_API_KEY"))
        clicked = st.button(
            build_btn,
            disabled=build_disabled,
            help=(build_help_off if build_disabled else build_help_on),
        )
        if clicked:
            with st.spinner(idx_spinner):
                vs = build_vectorstore(uploaded)
                st.session_state.vectorstore = vs
                st.session_state.chain = create_chain(vs)
            st.success(idx_done)

    st.session_state.setdefault("offline_msgs", [])
    st.session_state.setdefault("online_msgs", [])

    if not os.getenv("OPENAI_API_KEY"):
        st.info(offline_banner)
        for m in st.session_state.offline_msgs:
            with st.chat_message(m["role"]):
                if m.get("ts"):
                    st.caption(m["ts"])
                st.markdown(m["content"])
        user_q = st.chat_input(chat_ph_offline)
        if user_q:
            ts_now = now_ts(lang)
            st.session_state.offline_msgs.append({"role": "user", "content": user_q, "ts": ts_now})
            with st.chat_message("user"):
                st.caption(ts_now)
                st.markdown(user_q)
            ans = (small_talk_zh(user_q) if lang == "zh" else small_talk_response(user_q)) or offline_hint
            ts_ans = now_ts(lang)
            st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
            with st.chat_message("assistant"):
                st.caption(ts_ans)
                st.markdown(ans)
        st.stop()

    if os.getenv("OPENAI_API_KEY") and "chain" not in st.session_state:
        st.info(api_hint)
        for m in st.session_state.offline_msgs:
            with st.chat_message(m["role"]):
                if m.get("ts"):
                    st.caption(m["ts"])
                st.markdown(m["content"])
        tmp_q = st.chat_input(chat_ph_offline)
        if tmp_q:
            ts_now = now_ts(lang)
            st.session_state.offline_msgs.append({"role": "user", "content": tmp_q, "ts": ts_now})
            with st.chat_message("user"):
                st.caption(ts_now)
                st.markdown(tmp_q)
            ans = (small_talk_zh(tmp_q) if lang == "zh" else small_talk_response(tmp_q)) or build_help_off
            ts_ans = now_ts(lang)
            st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
            with st.chat_message("assistant"):
                st.caption(ts_ans)
                st.markdown(ans)
        st.stop()

    if "chain" in st.session_state:
        for m in st.session_state.online_msgs:
            with st.chat_message(m["role"]):
                if m.get("ts"):
                    st.caption(m["ts"])
                st.markdown(m["content"])
        user_q = st.chat_input(chat_ph_online)
        if user_q:
            ts_user = now_ts(lang)
            st.session_state.online_msgs.append({"role": "user", "content": user_q, "ts": ts_user})
            with st.chat_message("user"):
                st.caption(ts_user)
                st.markdown(user_q)

            smalltalk = small_talk_zh_basic(user_q) if lang == "zh" else small_talk_response_basic(user_q)
            if smalltalk is not None:
                final_md = smalltalk
            else:
                with st.spinner(ans_spinner):
                    try:
                        system_hint = (
                            "You are a helpful Tenant Assistant. Answer ONLY based on the uploaded documents. "
                            "If the answer isn't present in the documents, say you don't have enough information."
                            if lang == "en"
                            else "你是一名租客助手。仅根据已上传文档作答；若文档中没有答案，请说明信息不足。"
                        )
                        query = f"{system_hint}\nQuestion: {user_q}"
                        resp = st.session_state.chain.invoke({"question": query})
                        final_md = resp.get("answer", "（暂无答案）" if lang == "zh" else "(no answer)")
                    except Exception as e:
                        msg = str(e)
                        if "insufficient_quota" in msg or "429" in msg:
                            final_md = (
                                "（模型额度不足或达到速率限制，请更换可用模型或检查账单）"
                                if lang == "zh"
                                else "Quota/rate limit hit. Please switch to an available model or check billing."
                            )
                        elif "401" in msg or "invalid_api_key" in msg.lower():
                            final_md = "（API Key 无效，请在侧栏重新设置）" if lang == "zh" else "Invalid API key. Please set it in the sidebar."
                        else:
                            final_md = (f"（RAG 调用失败：{e}）" if lang == "zh" else f"RAG call failed: {e}")

            ts_ans = now_ts(lang)
            st.session_state.online_msgs.append({"role": "assistant", "content": final_md, "ts": ts_ans})
            with st.chat_message("assistant"):
                st.caption(ts_ans)
                st.markdown(final_md)

# --- page: repair ticket ---
elif st.session_state.page == "ticket":
    is_zh = st.session_state.get("lang", "en") == "zh"
    if is_zh:
        st.title("🧰 创建报修工单")
        issue_label = "问题标题"; issue_ph = "厨房水槽漏水"
        desc_label = "问题描述"; desc_ph = "请描述具体情况…"
        submit_btn = "📨 提交报修"
        created_ok = "报修已保存到数据库！"
        my_tickets = "我的报修工单"
        status_open = "进行中"
        empty_hint = "暂无工单"
        clear_btn = "🗑️ 清除所有报修记录"
    else:
        st.title("🧰 Create Repair Ticket")
        issue_label = "Issue title"; issue_ph = "Leaking sink in kitchen"
        desc_label = "Description";  desc_ph = "Describe the issue…"
        submit_btn = "📨 Submit Ticket"
        created_ok = "Ticket saved to database!"
        my_tickets = "My Tickets"
        status_open = "open"
        empty_hint = "No tickets yet"
        clear_btn = "🗑️ Clear All Tickets"

    # 提交表单
    with st.form("ticket_form", clear_on_submit=True):
        t_title = st.text_input(issue_label, placeholder=issue_ph)
        t_desc  = st.text_area(desc_label, placeholder=desc_ph)
        submitted = st.form_submit_button(submit_btn)
        if submitted:
            if not t_title.strip():
                st.warning("请填写问题标题。" if is_zh else "Please enter a title.")
            else:
                try:
                    new_id = create_ticket(t_title.strip(), t_desc.strip())
                    st.success(f"{created_ok}  (#{new_id})")
                except Exception as e:
                    st.error(f"DB error: {e}")

    st.subheader(my_tickets)

    # 先处理清空按钮，再读取列表
    if st.button(clear_btn, key="clear_all_tickets"):
        try:
            with closing(get_db_conn()) as conn:
                with conn, conn.cursor() as cur:
                    # 原来：cur.execute("DELETE FROM repair_tickets;")
                    cur.execute("TRUNCATE TABLE repair_tickets RESTART IDENTITY;")
            st.success("所有报修记录已删除！" if is_zh else "All tickets deleted!")
            st.rerun()
        except Exception as e:
            st.error(f"DB delete error: {e}")

    # 读取 & 渲染
    try:
        rows = list_reminders()
    except Exception as e:
        rows = []
        st.error(f"DB read error: {e}")

    if not rows:
        st.caption(empty_hint)
    else:
        for r in rows:
            # 直接格式化数据库时间（不做时区转换）
            ts_str = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            st.write(fmt_line.format(day=r["day_of_month"], note=r["note"] or "—"))
            st.caption(f"Created at: {ts_str} (SGT)")

# --- page: rent reminder ---
elif st.session_state.page == "reminder":
    is_zh = st.session_state.get("lang", "en") == "zh"
    if is_zh:
        st.title("💰 创建房租提醒")
        day_label = "每月几号"; note_label = "备注"; note_ph = "通过银行卡尾号••1234转账"
        save_btn = "💾 保存提醒"; saved_ok = "提醒已保存到数据库！"
        current_title = "当前提醒"
        fmt_line = "每月的第 **{day}** 天 — {note}"
        empty_hint = "暂无提醒"
        clear_btn = "🗑️ 清除所有提醒"
    else:
        st.title("💰 Create Rent Reminder")
        day_label = "Due day of month"; note_label = "Note"; note_ph = "Pay via bank transfer ending ••1234"
        save_btn = "💾 Save Reminder"; saved_ok = "Reminder saved to database!"
        current_title = "Current Reminder"
        fmt_line = "Every month on day **{day}** — {note}"
        empty_hint = "No reminders yet"
        clear_btn = "🗑️ Clear All Reminders"

    # 表单
    with st.form("reminder_form", clear_on_submit=True):
        r_day  = st.number_input(day_label, 1, 31, 1)
        r_note = st.text_input(note_label, placeholder=note_ph)
        r_submit = st.form_submit_button(save_btn)
        if r_submit:
            try:
                rid = create_reminder(int(r_day), (r_note or "").strip())
                st.success(f"{saved_ok}  (#{rid})")
            except Exception as e:
                st.error(f"DB error: {e}")

    st.subheader(current_title)

    # 先处理清空按钮，再读取列表
    if st.button(clear_btn, key="clear_all_reminders"):
        try:
            with closing(get_db_conn()) as conn:
                with conn, conn.cursor() as cur:
                    # 原来：cur.execute("DELETE FROM rent_reminders;")
                    cur.execute("TRUNCATE TABLE rent_reminders RESTART IDENTITY;")
            st.success("所有提醒已清空！" if is_zh else "All reminders deleted!")
            st.rerun()
        except Exception as e:
            st.error(f"DB delete error: {e}")

    # 读取 & 渲染
    try:
        rows = list_reminders()
    except Exception as e:
        rows = []
        st.error(f"DB read error: {e}")

    if not rows:
        st.caption(empty_hint)
    else:
        for r in rows:
            # 直接格式化数据库时间（不做时区转换）
            ts_str = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            st.write(fmt_line.format(day=r["day_of_month"], note=r["note"] or "—"))
            st.caption(f"Created at: {ts_str} (SGT)")
            
# --- page: offline chat ---
elif st.session_state.page == "offline":
    lang = st.session_state.get("lang", "en")
    if lang == "zh":
        st.title("💬 通用离线聊天")
        st.caption("无需 API，仅支持基础闲聊与引导。")
        history_empty_hint = "打个招呼或问一些基础问题…"
        offline_hint = "当前为离线聊天模式。你也可以在侧栏切换到“合同问答”。"
    else:
        st.title("💬 General Chat (Offline)")
        st.caption("No API required. Small talk and quick help only.")
        history_empty_hint = "Say hello or ask about some basic information…"
        offline_hint = (
            "I'm in offline chat mode. You can explore the sidebar features, "
            "or switch to Contract Chat for document-based Q&A."
        )

    if "offline_msgs" not in st.session_state:
        st.session_state.offline_msgs = []

    for m in st.session_state.offline_msgs:
        with st.chat_message(m["role"]):
            if m.get("ts"):
                st.caption(m["ts"])
            st.markdown(m["content"])

    user_q = st.chat_input(history_empty_hint)
    if user_q:
        ts_now = now_ts(lang)
        st.session_state.offline_msgs.append({"role": "user", "content": user_q, "ts": ts_now})
        with st.chat_message("user"):
            st.caption(ts_now)
            st.markdown(user_q)
        ans = (small_talk_zh(user_q) if lang == "zh" else small_talk_response(user_q)) or offline_hint
        ts_ans = now_ts(lang)
        st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
        with st.chat_message("assistant"):
            st.caption(ts_ans)
            st.markdown(ans)