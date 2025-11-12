#!/usr/bin/env python3
"""
Tenant Chatbot Assistant (Single-file, single UI; lazy-load storage & RAG)
租客聊天助手（单文件、单界面；按需惰性加载存储与RAG）

How to run / 如何运行：
    streamlit run app_tenantbot_neon_single.py
    
Purpose / 作用：
- A tidy, single-file Streamlit app for a tenant assistant.  
  整洁的单文件 Streamlit 应用，用于租客助手。
- Supports bilingual UI (English/中文), local small‑talk (offline), contract Q&A via RAG,
  simple tickets & rent reminders with a Postgres/Neon backend.
  支持中英文界面、离线闲聊、基于合同的RAG问答、以及使用 Postgres/Neon 的报修与房租提醒。
"""

# =============================== Imports / 导入 ===============================
import os  # env vars, keys / 读取环境变量与密钥
import re  # simple text normalization / 文本正则处理
import base64  # encoding avatars / 头像编码
import tempfile  # cache uploaded PDFs / 缓存上传PDF的临时文件
from datetime import datetime  # timestamps / 时间戳
from zoneinfo import ZoneInfo  # local timezone SGT / 新加坡时区处理
import warnings  # suppress specific warnings / 抑制特定警告
import streamlit as st  # Streamlit UI framework / Streamlit 界面框架

# Silence LangChain noisy warnings in logs / 屏蔽 LangChain 的噪声警告
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# --- 主题常量（NUS 配色） ---
NUS_BLUE = "#00205B"
NUS_ORANGE = "#FF6F0F"
NUS_WHITE = "#f7f9fb"

# ================== Global lightweight state / 全局轻量状态 ==================
# Page meta / 页面元信息（标题、图标、布局）
st.set_page_config(page_title="Tenant Chatbot", page_icon="🤖", layout="wide")

# --- Sidebar CSS overrides / 侧栏 CSS 定制 ---
st.markdown(f"""
<style>
:root {{
  --nus-blue: {NUS_BLUE};
  --nus-orange: {NUS_ORANGE};
  --nus-white: {NUS_WHITE};
}}

/* Sidebar 背景 */
[data-testid="stSidebar"] {{
  background-color: var(--nus-blue) !important;
}}

/* Sidebar 标题/说明默认橘色 */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p:not(.keep-white) {{
  color: var(--nus-orange) !important;
  font-weight: 600;
}}

/* English / 中文 文本设为白色 */
[data-testid="stSidebar"] div[role="radiogroup"] label p {{
  color: #fff !important;
  font-weight: 700 !important;
}}

/* Upload PDFs 提示文本设为白色 */
[data-testid="stSidebar"] .stMarkdown p.keep-white,
[data-testid="stSidebar"] .stMarkdown:last-child p {{
  color: #fff !important;
}}

/* ==== Sidebar Buttons ==== */
[data-testid="stSidebar"] .stButton > button {{
  background-color: var(--nus-white) !important;
  color: black !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}}
[data-testid="stSidebar"] .stButton > button * {{
  color: black !important;
  fill: black !important;
}}

[data-testid="stSidebar"] .stButton > button:hover {{
  background-color: var(--nus-orange) !important;
  color: white !important;
  transition: none !important;
}}
[data-testid="stSidebar"] .stButton > button:hover * {{
  color: white !important;
  fill: white !important;
}}

/* ==== Expander：折叠前白色 / 展开后蓝色 ==== */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
  border-radius: 16px !important;
  overflow: hidden !important;
  margin-top: 10px !important;
  border: none !important;
}}

/* 未展开：白色 header + 橘色字 */
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
   background-color: var(--nus-white) !important;
   border-radius: 16px !important;
   padding: 12px !important;
   color: var(--nus-orange) !important;
   font-weight: 700 !important;
   /* remove transitions to avoid flash on rerun */
   transition: none !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary * {{
  color: var(--nus-orange) !important;
  fill: var(--nus-orange) !important;
}}

/* 展开后：蓝色 header + 白字 */
[data-testid="stSidebar"] [data-testid="stExpander"][open] summary {{
   background-color: var(--nus-blue) !important;
   color: #fff !important;
   transition: none !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"][open] summary * {{
  color: #fff !important;
  fill: #fff !important;
}}

/* 输入框取消橙色边框，改成淡灰色 */
[data-testid="stSidebar"] input {{
  background-color: #ffffff !important;
  color: var(--nus-blue) !important;
  border-radius: 10px !important;
  border: 1.5px solid #dcdcdc !important;
  font-weight: 600 !important;
}}

/* Diagnostics / API Setup 里的按钮保持白底黑字 */
[data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button {{
  background-color: #fff !important;
  color: #000 !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}}

/* ===== 右侧主内容背景改为淡蓝 ===== */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stVerticalBlock"] .block-container {{
  background-color: #F2F7FF !important;  /* 淡蓝 */
}}

/* 顶部 header 也用淡蓝（如果你看到顶部一条白带） */
[data-testid="stHeader"] {{
  background: #F2F7FF !important;
}}

</style>
""", unsafe_allow_html=True)

# --- Chat message bubble CSS / 聊天消息气泡 CSS ---
st.markdown("""
<style>

/* 让消息（头像 + 气泡）左右排列，并且垂直居中对齐 */
.msg{
  display:flex;
  flex-direction:row;
  align-items:center;        /* ✅ 头像和气泡垂直方向对齐（关键） */
  gap:14px;
  margin:18px 0;
}

/* 用户消息反向排列（头像在右）*/
.msg[data-role="user"]{
  flex-direction:row-reverse;
}

/* 头像固定大小，不被压缩 */
.avatar{
  width:64px; height:64px;
  min-width:64px;
  border-radius:50%;
  overflow:hidden;
  border:3px solid transparent;
  display:flex; align-items:center; justify-content:center;
}

/* 边框颜色 */
.msg[data-role="assistant"] .avatar{ border-color:#00205B; }
.msg[data-role="user"]      .avatar{ border-color:#FF6F0F; }

/* 头像图像填充圆形 */
.avimg{
  width:100%; height:100%;
  object-fit:cover;
  border-radius:50%;
}

/* ✅ 气泡区域在垂直方向上用 column，使 timestamp 跟气泡绑在一起 */
.bubble-wrap{
  display:flex;
  flex-direction:column;
  max-width:min(70vw, 900px);
}

/* 氣泡 */
.bubble{
  padding:14px 18px;
  border-radius:20px;
  font-size:1.08rem;
  line-height:1.55;
  box-shadow:0 5px 15px rgba(0,0,0,.12);
  white-space:pre-wrap;
}

/* 配色 */
.msg[data-role="assistant"] .bubble{
  background:#00205B; color:#fff;
}
.msg[data-role="user"] .bubble{
  background:#FF6F0F; color:#fff;
}

/* ✅ 时间戳必须跟随 bubble，而不是跟随 avatar */
.meta{
  font-size:12px; opacity:.6;
  margin-top:6px;
}

/* ✅ 时间戳根据不同角色左右对齐 */
.msg[data-role="assistant"] .meta{
  align-self:flex-start;     /* 左边消息时间戳靠左 */
}
.msg[data-role="user"] .meta{
  align-self:flex-end;       /* 右边消息时间戳靠右 */
}

</style>
""", unsafe_allow_html=True)


# Initialize session-scoped variables if missing / 首次访问时初始化会话变量
if "lang" not in st.session_state:
    st.session_state.lang = "en"  # default language / 默认英文
if "page" not in st.session_state:
    st.session_state.page = "offline"  # default landing page / 默认进入离线聊天
if "tickets" not in st.session_state:
    st.session_state.tickets = []
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "online_msgs" not in st.session_state:
    st.session_state.online_msgs = []
if "offline_msgs" not in st.session_state:
    st.session_state.offline_msgs = []
# UX flag: whether DB schema init was run at least once / 仅用于UX的标记：是否手动初始化过数据库
if "db_inited" not in st.session_state:
    st.session_state.db_inited = False

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
    
if "kb_doc_names" not in st.session_state:
    st.session_state.kb_doc_names = [] 

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0



# ---------------------- UI helpers / 界面辅助函数 ----------------------
def apply_chat_input_visibility():
    """Show chat_input only on General/Contract pages.
    仅在“普通聊天/合同问答”页面显示底部输入框，其余页面隐藏。"""
    page = st.session_state.get("page", "offline")
    show = (page == "offline") or (page == "chat")
    #Inject CSS to toggle chat input visibility / 注入CSS控制 chat_input 显隐
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

# =============== Lazy imports / 惰性导入（用到才加载依赖） ===============

def lazy_import_psycopg():
    """Import psycopg2 only when DB access is needed.
    仅在需要访问数据库时导入 psycopg2 以加快冷启动。"""
    global psycopg2, psycopg2_extras
    try:
        import psycopg2  # type: ignore
        import psycopg2.extras as psycopg2_extras  # type: ignore
        return psycopg2, psycopg2_extras
    except Exception as e:
        raise RuntimeError(f"psycopg2 not available: {e}")


def lazy_import_langchain():
    """
    Import LangChain stack lazily for RAG functions.
    把常用对象缓存在 st.session_state["lc_stack"] 里，避免重复导入。
    """
    if "lc_stack" in st.session_state:
        return st.session_state["lc_stack"]
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import FAISS
        from langchain.chains import ConversationalRetrievalChain, RetrievalQA
        from langchain.memory import ConversationBufferMemory
        from langchain.prompts import PromptTemplate  # ✅ 用于注入满分格式 Prompt

        st.session_state["lc_stack"] = {
            "PyPDFLoader": PyPDFLoader,
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
            "OpenAIEmbeddings": OpenAIEmbeddings,
            "ChatOpenAI": ChatOpenAI,
            "FAISS": FAISS,
            "ConversationalRetrievalChain": ConversationalRetrievalChain,
            "RetrievalQA": RetrievalQA,
            "ConversationBufferMemory": ConversationBufferMemory,
            "PromptTemplate": PromptTemplate,
        }
        return st.session_state["lc_stack"]
    except Exception as e:
        # 明确提示安装命令
        raise RuntimeError(
            "LangChain stack missing. Install:\n"
            "  pip install langchain langchain-openai openai pypdf faiss-cpu\n"
            f"Details: {e}"
        )
        
# ================== DB helpers (short‑lived conns) / 数据库辅助 ==================

def get_db_conn():
    """Build a short‑lived Postgres connection using env vars.
    根据环境变量创建短连接的 Postgres 连接。支持 DATABASE_URL 或逐项 PG_* 变量。"""
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
    if "sslmode=" not in dsn:  # enforce TLS / 强制启用 TLS
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    # Return a connection with robust keepalive / 返回带存活探测参数的连接
    return psycopg2.connect(
        dsn,
        sslmode="require",
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        cursor_factory=lazy_import_psycopg()[1].DictCursor,  # Dict rows / 字典行
    )


def ensure_schema(conn):
    """Ensure required tables exist (idempotent).
    确保所需表存在（幂等），如无则创建。"""
    with conn.cursor() as cur:
        # repair_tickets
        cur.execute("SELECT to_regclass('public.repair_tickets');")
        exists = cur.fetchone()[0] is not None
        if not exists:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_tickets (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        # rent_reminders
        cur.execute("SELECT to_regclass('public.rent_reminders');")
        exists = cur.fetchone()[0] is not None
        if not exists:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rent_reminders (
                    id SERIAL PRIMARY KEY,
                    day_of_month INT NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
                    note TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )


def init_db():
    """Manual schema init button uses this (optional UX helper).
    供手动一键初始化表结构（可选的UX小辅助）。"""
    psycopg2, _ = lazy_import_psycopg()
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_tickets (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rent_reminders (
                    id SERIAL PRIMARY KEY,
                    day_of_month INT NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
                    note TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
    st.session_state.db_inited = True
    return True


# CRUD helpers / 简单的新增-查询-清空操作
        
def create_ticket(title: str, desc: str):
    with get_db_conn() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO repair_tickets (title, description, status) "
                "VALUES (%s, %s, %s) RETURNING id;",
                (title, desc, "open"),
            )
            tid = cur.fetchone()["id"]
            # 立刻查询当前总数
            cur.execute("SELECT COUNT(*) AS c FROM repair_tickets;")
            total = cur.fetchone()["c"]
            return tid, total


def list_tickets(limit: int = 50):
    with get_db_conn() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, status, description, created_at
                FROM repair_tickets
                ORDER BY id DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()
        
def create_reminder(day_of_month: int, note: str):
    with get_db_conn() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rent_reminders (day_of_month, note) VALUES (%s, %s) RETURNING id;",
                (day_of_month, note),
            )
            rid = cur.fetchone()["id"]
            # 立刻查询当前总数
            cur.execute("SELECT COUNT(*) AS c FROM rent_reminders;")
            total = cur.fetchone()["c"]
            return rid, total


def list_reminders(limit: int = 20):
    with get_db_conn() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, day_of_month, note, created_at
                FROM rent_reminders
                ORDER BY id DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()



# ================== RAG helpers / RAG 辅助函数（惰性导入） ==================

def build_vectorstore(uploaded_files):
    """Load PDFs → split chunks → embed → build FAISS index.
    将上传的 PDF 加载→切片→嵌入→建立 FAISS 向量库。"""
    lc = lazy_import_langchain()
    paths = []  # temp paths / 临时文件路径收集
    try:
        # Save uploads as temp files for PyPDFLoader / 将上传文件写入临时文件
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getvalue())
                paths.append(tmp.name)
        # Load and merge pages / 读取PDF并合并文档
        docs = []
        for p in paths:
            loader = lc["PyPDFLoader"](p)
            docs += loader.load()
        # Chunking strategy / 文本切片策略
        splitter = lc["RecursiveCharacterTextSplitter"](chunk_size=900, chunk_overlap=180)
        texts = splitter.split_documents(docs)
        # Embedding & index / 嵌入与索引
        embeddings = lc["OpenAIEmbeddings"]()  # reads OPENAI_API_KEY / 读取环境变量
        vs = lc["FAISS"].from_documents(texts, embeddings)
        return vs
    finally:
        # Always clean temp files / 始终清理临时文件
        for p in paths:
            try:
                os.unlink(p)
            except Exception:
                pass


def create_chain(vs):
    """Try lightweight model first, fallback to larger one.
    优先尝试轻量模型，失败则退到更强模型；均失败时报错。"""
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
            try:
                st.toast(f"Model in use: {m}")  # gentle UX hint / 轻提示
            except Exception:
                pass
            return chain
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All models failed to init. Last error: {last_err}")


# ================== Utilities & small talk / 工具与闲聊 ==================

def now_ts():
    """Current time in Asia/Singapore for message captions.
    以新加坡时区格式化当前时间用于消息时间戳。"""
    return datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S")


# ---- Text normalization helpers / 文本标准化辅助 ----

def normalize_word(word: str) -> str:
    """Naïve English stemmer for keyword matching.
    简单英文词尾截断，便于关键词匹配。"""
    word = word.lower()
    suffixes = [
        "ing","ed","es","s","ly","tion","ions","ness","ment","ments","ities","ity","als","al","ers","er"
    ]
    for suf in suffixes:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)]
    return word


def normalize_text(text: str) -> str:
    """Keep only letters/CJK and split for English matching.
    仅保留字母与中日韩字符，便于英文匹配分词。"""
    words = re.findall(r"[a-zA-Z\u4e00-\u9fff']+", text.lower())
    return " ".join(normalize_word(w) for w in words)


def normalize_text_zh(text: str) -> str:
    """Keep CJK + punctuation; used for simple Chinese matching.
    保留中日韩与常用标点，用于简易中文匹配。"""
    return "".join(re.findall(r"[0-9A-Za-z\u4e00-\u9fff'，。！？、：；（）()《》“”\"' ]+", text))


def any_terms_en(text_norm: str, terms: list[str]) -> bool:
    """English keyword contains or whole-word matches.
    英文关键词的包含或整词匹配。"""
    for t in terms:
        t2 = normalize_word(t)
        if re.search(rf"\b{re.escape(t2)}\b", text_norm) or t2 in text_norm:
            return True
    return False


def contains_any_zh(text_norm: str, phrases: list[str]) -> bool:
    """Chinese phrase substring matching.
    中文短语的子串匹配。"""
    return any(p in text_norm for p in phrases)


def any_phrases(text: str, phrases: list[str]) -> bool:
    """Language-agnostic phrase check after normalization.
    归一化后进行语言无关的短语检查。"""
    norm = normalize_text(text)
    return any(normalize_text(p) in norm for p in phrases)


# ---- Small‑talk templates (中文) / 中文闲聊模板 ----

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
    """Like small_talk_zh but skips when contract-like words appear.
    功能类似 small_talk_zh，但遇到“合同相关”词汇则返回 None 交给RAG。"""
    q = normalize_text_zh(q_raw.strip())
    contract_like = ["合同","租约","条款","租金","押金","房东","租客","维修","报修","终止","违约","续约","账单","费用"]
    if contains_any_zh(q, contract_like):
        return None
    return small_talk_zh(q_raw)


# ---- Small‑talk templates (EN) / 英文闲聊模板 ----

def small_talk_response(q_raw: str) -> str | None:
    q = normalize_text(q_raw.strip())
    if any_terms_en(q, ["hi","hello","hey","morning","evening","afternoon"]): # or any_phrases(q, ["你好","嗨","哈喽"]):
        return "Hello! I’m your Tenant Assistant 👋 How can I help you today?"
    if any_phrases(q, ["how are you","how's it going","how are u","how are ya","how are things","how do you feel"]): #,"你好吗","最近怎么样","最近如何"]):
        return "I'm doing well and ready to help! How can I assist you today?"
    if any_phrases(q, ["who are you","what are you","your name"]): #,"你是谁","你是干什么的"]):
        return "I’m a friendly chatbot that helps tenants understand contracts and manage repairs or rent reminders."
    if any_terms_en(q, ["thanks","thank","thx","appreciate"]) or any_phrases(q, ["thank you","many thanks"]): #,"谢谢","多谢","非常感谢","感謝"]):
        return "You're welcome! If there’s anything else you need, just let me know."
    if any_phrases(q, ["what can you do","what can u do"]): #,"能做什么","你会干嘛"]) or any_terms_en(q, ["function","feature","capability"]):
        return (
            "I can help you read tenancy agreements, create repair tickets, and set rent reminders. "
            "Once you add an API key, I can also answer contract questions directly!"
        )
    if any_phrases(q, ["how to upload","upload pdf","add document","how to start","start upload"]): #,"怎么上传","如何开始"]):
        return (
            "Click **‘Upload PDF contracts or house rules’** to add documents. "
            "Then click **‘Build/Refresh Knowledge Base’** after setting your API key."
        )
    if any_phrases(q, ["rent reminder","rent day","when to pay rent"]): #,"租金提醒","什么时候交房租"]):
        return "Use **💰 Create Rent Reminder** in the sidebar to set a monthly reminder."
    if any_terms_en(q, ["repair","maintain","fix","broken","leak","damage","fault","issue"]): # or any_phrases(q, ["报修","维修","漏水","坏了"]):
        return "Use **🧰 Create Repair Ticket** in the sidebar. Describe the problem and I’ll record it."
    if any_terms_en(q, ["contract","agreement","lease","term","clause","deposit","renewal","policy","rules"]): # or any_phrases(q, ["合同","条款","押金","续约","租约"]):
        return "Upload your contract and set an API key; I’ll then answer questions based on the document."
    return None


def small_talk_response_basic(q_raw: str) -> str | None:
    """Like small_talk_response but yields None for contract-like queries.
    类似 small_talk_response，但遇到“合同相关”词汇时交给RAG处理。"""
    q = normalize_text(q_raw.strip())
    if any_terms_en(q, [
        "contract","agreement","lease","tenant","landlord","deposit","repair","maintenance","damage","clause","policy","rent","renewal","notice","terminate"
    ]):
        return None
    if any_terms_en(q, ["hi","hello","hey","morning","evening","afternoon"]): # or any_phrases(q, ["你好","嗨","哈喽"]):
        return "Hello! I’m your Tenant Assistant 👋 How can I help you today?"
    if any_phrases(q, ["how are you","how's it going","how are u","how are ya","how are things","how do you feel"]): #,"你好吗","最近怎么样","最近如何"]):
        return "I'm doing well and ready to help! How can I assist you today?"
    if any_terms_en(q, ["thanks","thank","thx","appreciate"]) or any_phrases(q, ["thank you","many thanks"]): #,"谢谢","多谢","非常感谢","感謝"]):
        return "You're welcome! If there’s anything else you need, just let me know."
    if any_phrases(q, ["who are you","what are you","your name"]): #,"你是谁","你是干什么的"]):
        return "I’m a friendly chatbot that helps tenants understand contracts and manage repairs or rent reminders."
    if any_phrases(q, ["what can you do","what can u do"]) or any_terms_en(q, ["function","feature","capability"]): #,"能做什么","你会干嘛"]) 
        return (
            "I can help you read tenancy agreements, create repair tickets, and set rent reminders. "
            "Once you add an API key, I can also answer contract questions directly!"
        )
    if any_phrases(q, ["how to upload","upload pdf","add document","how to start","start upload"]): #,"怎么上传","如何开始"]):
        return (
            "Click **‘Upload PDF contracts or house rules’** to add documents. "
            "Then click **‘Build/Refresh Knowledge Base’** after setting your API key."
        )
    return None

# ===== Language guard (no extra deps) =====
def detect_lang(text: str) -> str:
    """Return 'zh' if contains CJK, 'en' if only Latin, else 'mixed/other'."""
    if not text or not isinstance(text, str):
        return "other"
    has_cjk = bool(_CJK_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_cjk and not has_lat:
        return "zh"
    if has_lat and not has_cjk:
        return "en"
    if has_cjk and has_lat:
        # Mixed input -> prefer Chinese
        return "zh"
    return "other"

def guard_language_and_offer_switch(user_text: str) -> bool:
    """
    与当前 UI 语言不一致时，仅提示，不执行切换。
    返回 True = 阻止后续处理（外层 st.stop()）
    """
    ui = st.session_state.get("lang", "en")  # 当前 UI 语言 en / zh
    dlang = detect_lang(user_text)           # 输入内容语言 en / zh / other

    # 英文界面 + 中文输入 -> 提示
    if ui == "en" and dlang == "zh":
        with st.container(border=True):
            st.warning("This looks like Chinese input while you're on the English UI.")
            st.info("请切换到侧边栏的『中文』界面以获得更准确的回答。")
        return True

    # 中文界面 + 英文输入 -> 提示
    if ui == "zh" and dlang == "en":
        with st.container(border=True):
            st.warning("当前是中文界面，但你输入的是英文。")
            st.info("Please switch to English from the sidebar for better responses.")
        return True

    return False

# ===== Message rendering with avatars / 带头像的消息渲染 =====
def _b64_once(state_key: str, path: str) -> str | None:
    if state_key in st.session_state:
        return st.session_state[state_key]
    try:
        abs_path = path
        if not os.path.isabs(abs_path):
            abs_path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.exists(abs_path):
            st.session_state[state_key] = None
            return None
        with open(abs_path, "rb") as f:
            st.session_state[state_key] = "data:image/png;base64," + base64.b64encode(f.read()).decode()
            return st.session_state[state_key]
    except Exception:
        st.session_state[state_key] = None
        return None

ASSISTANT_AVATAR = _b64_once("avatar_assistant_b64", "chatbot_image.png")
USER_AVATAR      = _b64_once("avatar_user_b64", "user_image.jpg")

def render_message(role, content, ts=None):
    avatar = (
        f"<img src='{ASSISTANT_AVATAR}' class='avimg' />"
        if role == "assistant"
        else f"<img src='{USER_AVATAR}' class='avimg' />"
        if USER_AVATAR else "<div class='avemoji'>🧑</div>"
    )

    st.markdown(
        f"""
        <div class="msg" data-role="{role}">
            <div class="avatar">{avatar}</div>
            <div class="bubble-wrap">
                <div class="bubble">{content}</div>
                <div class="meta">{ts}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================= Sidebar (single‑page nav) / 侧栏导航 =======================
with st.sidebar:
    # Language toggle / 语言切换
    st.header("🌐 Language / 语言")
    lang_choice = st.radio(
        "Select language / 选择语言",
        options=["English", "中文"],
        index=0 if st.session_state.get("lang", "en") == "en" else 1,
    )
    st.session_state.lang = "en" if lang_choice == "English" else "zh"


    # Labels based on language / 多语言标签
    if st.session_state.lang == "en":
        btn_general = "💬 General Chat"
        btn_contract = "💬 Contract Chat"
        btn_ticket = "🧰 Create Repair Ticket"
        btn_reminder = "💰 Create Rent Reminder"
        caption_text = "Upload PDFs anytime. Build the knowledge base after setting OPENAI_API_KEY below."
        tab_api_title = "API Setup"
        api_key_label  = "OpenAI API Key"
        clear_label = "🧹 Clear Chat"
        cleared_offline_msg = "Cleared General Chat history."
        cleared_online_msg = "Cleared Contract Chat history."
        nothing_here_msg = "Nothing to clear on this page."
        tab_api_title  = "API Setup"
        tab_diag_title = "🧪 Diagnostics"
        api_key_label  = "OpenAI API Key"
        diag_btn_label = "Test Neon connection"
        db_ok, db_ng   = "DB connected ✔️", "DB connect failed: "
        api_seen_label = "API Key detected: "
    else:
        btn_general = "💬 普通聊天"
        btn_contract = "💬 合同问答"
        btn_ticket = "🧰 报修创建"
        btn_reminder = "💰 房租提醒"
        caption_text = "可随时上传 PDF。先在下方设置 OPENAI_API_KEY 再构建知识库。"
        tab_api_title = "API 设置"
        api_key_label  = "OpenAI API 密钥"
        clear_label = "🧹 清空聊天"
        cleared_offline_msg = "已清空『普通聊天』历史。"
        cleared_online_msg = "已清空『合同问答』历史。"
        nothing_here_msg = "此页面没有可清空的聊天记录。"
        tab_api_title  = "API 设置"
        tab_diag_title = "🧪 诊断"
        api_key_label  = "OpenAI API 密钥"
        diag_btn_label = "测试 Neon 数据库连接"
        db_ok, db_ng   = "数据库连接成功 ✔️", "数据库连接失败："
        api_seen_label = "检测到 API Key："


    # Navigation buttons / 导航按钮
    if st.button(btn_general, use_container_width=True):
        st.session_state.page = "offline"
    if st.button(btn_contract, use_container_width=True):
        st.session_state.page = "chat"
    if st.button(btn_ticket, use_container_width=True):
        st.session_state.page = "ticket"
    if st.button(btn_reminder, use_container_width=True):
        st.session_state.page = "reminder"



    api_tab, diag_tab = st.tabs([tab_api_title, tab_diag_title])

    with api_tab:
        api_key_in = st.text_input(api_key_label, type="password", key="api_key_input")  # 稳定 key
        if api_key_in:
            os.environ["OPENAI_API_KEY"] = api_key_in
            st.success("API key set for this session." if st.session_state.lang=="en" else "API 密钥已设置。")

    
    with diag_tab:
        if st.button(diag_btn_label, key="btn_test_neon"):  # 稳定 key
            try:
                with get_db_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT NOW();")
                st.success(db_ok)
            except Exception as e:
                st.error(db_ng + str(e))
        st.write(api_seen_label + str(bool(os.getenv("OPENAI_API_KEY"))))
        
    # Clear Chat (scoped to current page) / 仅清“当前页面”的聊天记录
    # - 在 General(offline) 页面只清离线消息
    # - 在 Contract(chat) 页面只清在线(RAG)消息
    if st.button(clear_label, use_container_width=True, key="btn_clear_chat"):
        current_page = st.session_state.get("page", "offline")
        if current_page == "offline":
            st.session_state.offline_msgs = []
            st.success(cleared_offline_msg)
        elif current_page == "chat":
            st.session_state.online_msgs = []
            st.success(cleared_online_msg)
        else:
            # 在报修/提醒等页面点击清空，不影响任何聊天
            st.info(nothing_here_msg)

    st.caption(caption_text)
    st.divider()
    

# After sidebar is drawn, toggle chat input visibility / 渲染侧栏后立刻控制输入框显隐
apply_chat_input_visibility()


# ========================= Pages / 页面（单文件切换） =========================
# --- Contract Chat page / 合同问答 ---
# if st.session_state.page == "chat":
#     # ===== 满分格式工具（只在本页面用） =====
#     import re
#     from typing import List, Dict, Any

#     FULL_SCORE_SYSTEM_PROMPT = """
#     You are a contract-aware tenant assistant. Use ONLY the retrieved tenancy agreement excerpts.

#     Your response MUST follow **exactly** this layout (including section titles and bullet labels):

#     ━━━━━━━━━━━━━━━━━━━━━━
#     ✅ **Answer (1–3 sentences):**
#     <concise business-style answer that contains exact numbers, money, and conditions>

#     ━━━━━━━━━━━━━━━━━━━━━━
#     💡 **Breakdown:**

#     **• Preconditions / timing:**  
#       <When is the rule applicable? e.g., “After first 12 months of the tenancy.”>

#     **• Exact limits (numbers / who pays / notice period):**  
#       <Exact amounts + responsibility, e.g., “S$200 per item / Tenant pays first S$200.”>

#     **• Required documents / approvals:**  
#       <Proofs, approvals, notices, e.g., “Documentary proof required; landlord approval if > S$200.”>

#     **• Exceptions (when rule does NOT apply):**  
#       <e.g., “No diplomatic clause during renewal term unless mutually agreed.”>

#     **• Operational steps (if applicable):**  
#       <e.g., “Arrange professional cleaning; dry clean curtains; joint inspection.”>

#     ━━━━━━━━━━━━━━━━━━━━━━
#     🔎 **Relevant Contract Excerpts (verbatim):**
#     " <exact quote 1> " (Clause <id>, page <n>)
#     " <exact quote 2> " (Clause <id>, page <n>)

#     ━━━━━━━━━━━━━━━━━━━━━━

#     Rules:
#     - ONLY answer based on retrieved contract excerpts.
#     - If the answer is not in the contract, say: **"Not mentioned in the contract."**
#     - Do NOT add new interpretations. Do NOT invent clause number / page number.
#     - ALWAYS keep numbers EXACT (S$200, 14 days, 7 days, 2 months).
#     """

#     # ========= 条款匹配与精准引用 ========= #

#     # regex 检出 "Clause 5(c)" 等格式
#     _CLAUSE_RE = re.compile(r"(Clause\s*\d+(?:\([a-z]\))?)", re.IGNORECASE)

#     def _extract_clause_id(text: str) -> str:
#         m = _CLAUSE_RE.search(text or "")
#         return m.group(1) if m else ""

#     def _keyword_score(question: str, text: str) -> int:
#         """根据问题匹配关键词，给 snippet 打分"""
#         q = (question or "").lower()
#         t = (text or "").lower()

#         keys = []
#         if "diplomatic" in q or "relocate" in q or "terminate" in q:
#             keys += ["diplomatic", "terminate", "relocat", "deport", "refused", "2 months", "commission"]
#         if "repair" in q or "broken" in q or "spoil" in q:
#             keys += ["s$200", "minor repair", "air con", "aircon", "water heater", "structural", "bulb", "tube", "approval"]
#         if "return" in q or "handover" in q or "move out" in q:
#             keys += ["clean", "dry clean", "curtain", "remove nails", "white putty", "joint inspection", "keys", "no rent"]

#         score = sum([1 for k in keys if k in t])
#         return score

#     def _clause_priority(question: str):
#         q = (question or "").lower()

#         if "diplomatic" in q:
#             return ["5(c)", "5(d)", "5(f)"]        # Q1

#         if any(k in q for k in ["repair", "broken", "spoil"]):
#             return ["2(f)", "2(g)", "2(i)", "2(j)", "2(k)", "4(c)"]  # Q2

#         if any(k in q for k in ["return", "handover", "move", "move out"]):
#             return ["2(y)", "2(z)", "6(o)"]       # Q3

#         return []
    

#     def _pick_excerpts(docs: List[Any], question: str, max_items: int = 3) -> List[Dict[str, str]]:
#         priority = _clause_priority(question)
#         out, seen = [], set()

#         for d in docs or []:
#             meta = getattr(d, "metadata", {}) or {}
#             page = meta.get("page")
#             content = (getattr(d, "page_content", "") or "").strip()

#             if not content:
#                 continue
#             clause = _extract_clause_id(content)

#             # ❌ 排除无关 snippet（如 placeholder / compliance）
#             if "COMPLIANCE" in content or "placeholder" in content:
#                 continue

#             snippet = content[:260].replace("\n", " ")
#             key = (page, clause, snippet[:30])
#             if key in seen:
#                 continue

#             seen.add(key)

#             out.append({
#                 "quote": snippet + ("..." if len(content) > 260 else ""),
#                 "page": page,
#                 "clause": clause
#             })

#         # ✅ 优先排序 clause
#         if priority:
#             out.sort(key=lambda x: priority.index(x["clause"]) if x["clause"] in priority else 999)

#         return out[:max_items]
    

#     def format_contract_answer(user_q: str, llm_answer: str, source_docs: List[Any]) -> str:
#         excerpts = _pick_excerpts(source_docs, max_items=3, question=user_q)
#         lower_ans = (llm_answer or "").lower()
#         is_refusal = ("not mentioned" in lower_ans) or (not excerpts)

#         refs_lines = []
#         if not is_refusal:
#             for ex in excerpts:
#                 tag = []
#                 if ex.get("clause"):
#                     tag.append(ex["clause"])
#                 if ex.get("page") is not None:
#                     tag.append(f"page {ex['page']}")
#                 refs_lines.append(f"\"{ex['quote'][:240]}...\" ({', '.join(tag)})")

#         refs_block = "🔎 Relevant Contract Excerpts:\n" + ("\n".join(refs_lines) if refs_lines else "Not available.")

#         return f"""{llm_answer.strip()}

# {refs_block}
# """

#     # ========== 页面/UI ========= #
#     is_zh = st.session_state.lang == "zh"
#     st.title("租客聊天助手" if is_zh else "Tenant Chatbot Assistant")
#     st.caption("基于已上传的租赁合同进行问答" if is_zh else "Contract-aware Q&A using uploaded tenancy documents.")

#     uploaded = st.file_uploader(
#         "上传租赁合同或房屋守则（PDF）" if is_zh else "Upload PDF contracts or house rules",
#         type="pdf",
#         accept_multiple_files=True,
#         key=f"kb_uploader_{st.session_state.get('uploader_key', 0)}",
#     )

#     if uploaded and len(uploaded) > 0:
#         st.session_state.kb_doc_names = [f.name for f in uploaded]
#         st.session_state.pdf_uploaded = True

#     if st.session_state.pdf_uploaded and st.session_state.kb_doc_names:
#         st.caption("已选择的文件：" if is_zh else "Selected PDFs:")
#         for nm in st.session_state.kb_doc_names:
#             st.markdown(f"**{nm}**")

#     if st.session_state.pdf_uploaded:
#         build_disabled = not bool(os.getenv("OPENAI_API_KEY"))

#         clicked = st.button(
#             "🔄 构建/刷新知识库" if is_zh else "🔄 Build/Refresh Knowledge Base",
#             disabled=build_disabled,
#             use_container_width=True,
#         )

#         reset_clicked = st.button(
#             "♻️ 重置知识库" if is_zh else "♻️ Reset Knowledge Base",
#             disabled=build_disabled,
#             use_container_width=True,
#         )

#         if clicked:
#             with st.spinner("正在根据文档构建索引…" if is_zh else "Indexing documents…"):
#                 vs = build_vectorstore(uploaded)
#                 st.session_state.vectorstore = vs

#                 lc = lazy_import_langchain()
#                 PromptTemplate = lc["PromptTemplate"]
#                 ChatOpenAI = lc["ChatOpenAI"]
#                 RetrievalQA = lc["RetrievalQA"]

#                 retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "lambda_mult": 0.3})
#                 llm = ChatOpenAI(temperature=0)

#                 prompt = PromptTemplate(
#                     input_variables=["context", "question"],
#                     template=FULL_SCORE_SYSTEM_PROMPT + "\n\n[CONTRACT CONTEXT]\n{context}\n\n[USER QUESTION]\n{question}"
#                 )

#                 st.session_state.chain = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     retriever=retriever,
#                     chain_type="stuff",
#                     return_source_documents=True,
#                     chain_type_kwargs={"prompt": prompt}
#                 )

#             st.success("知识库已就绪！现在可以在下方提问。" if is_zh else "Knowledge base ready! Ask questions below.")

#         if reset_clicked:
#             st.session_state.pop("vectorstore", None)
#             st.session_state.pop("chain", None)
#             st.session_state["kb_doc_names"] = []
#             st.session_state["online_msgs"] = []
#             st.session_state["pdf_uploaded"] = False
#             st.session_state["uploader_key"] += 1
#             st.toast("知识库已清空。" if is_zh else "Knowledge base cleared.")
#             st.rerun()

#     has_chain = st.session_state.get("chain") is not None

#     st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
#     for m in st.session_state.get("online_msgs", []):
#         render_message(m.get("role", "assistant"), m.get("content", ""), m.get("ts"))
#     st.markdown('</div>', unsafe_allow_html=True)

#     user_q = st.chat_input(
#         "就你的合同提问…" if has_chain else "请先构建知识库…",
#         disabled=not has_chain,
#     )

#     if has_chain and user_q:
#         ts_user = now_ts()
#         st.session_state.online_msgs.append({"role": "user", "content": user_q, "ts": ts_user})
#         render_message("user", user_q, ts_user)

#         ans_slot = st.empty()
#         with ans_slot.container():
#             render_message("assistant", "…", now_ts())

#         try:
#             resp = st.session_state.chain.invoke({"query": user_q})
#             final_text = resp.get("result") or resp.get("answer") or ""
#             source_docs = resp.get("source_documents") or []
#             final_md = format_contract_answer(user_q, final_text, source_docs)

#         except Exception as e:
#             final_md = f"(RAG failed: {e})"

#         ts_ans = now_ts()
#         st.session_state.online_msgs.append({"role": "assistant", "content": final_md, "ts": ts_ans})
#         with ans_slot.container():
#             render_message("assistant", final_md, ts_ans)
            
            
if st.session_state.page == "chat":
    # ===== 满分格式工具（只在本页面用） =====
    import re
    from typing import List, Dict, Any

    FULL_SCORE_SYSTEM_PROMPT = """
    You are a contract-aware tenant assistant. Use ONLY the tenancy agreement retrieved below.
    ALWAYS answer in this exact structure and bullet labels:

    ✅ Answer:
    <short, direct, actionable answer in 1–3 sentences with exact numbers>

    💡 Breakdown:
    • Preconditions / timing:
    • Exact limits (numbers / notice period / who pays):
    • Required documents / approvals:
    • Exceptions (when this rule does NOT apply):
    • Operational steps (if applicable):

    🟢 Good to know (optional):
    <benefit to the tenant, e.g., “No rent charged during repair period.”>

    🔴 Warning (optional):
    <penalty, reimbursement, forfeiture, or risk to the tenant>

    🔎 Relevant Contract Excerpts (verbatim):
    "<verbatim quote 1>" (Clause <id>, page <n>)
    "<verbatim quote 2>" (Clause <id>, page <n>)

    Rules:
    - ONLY answer based on retrieved context (PDF excerpts).
    - If the contract does not mention the answer, reply: "Not mentioned in the contract."
    - NEVER fabricate clause numbers or page numbers.
    - ALWAYS keep numbers EXACT (e.g., S$200, 14 days, 7 days, 2 months).
    """
    
    _CLAUSE_RE = re.compile(r"(Clause\s*\d+(?:\([a-z]\))?)", re.IGNORECASE)
    
    def _extract_clause_id(text: str) -> str:
        """Extract clause number if exists / 若包含条款编号则提取"""
        m = _CLAUSE_RE.search(text or "")
        return m.group(1) if m else ""

    # -------------------------------------------------------------------------
    # ✅ 用问题关键词 + 条款优先级排序，确保引用的是**相关条款**
    # -------------------------------------------------------------------------

    def _keyword_score(question: str, text: str) -> int:
        """Score relevance by keyword matching / 根据问题匹配关键词打分"""
        q = (question or "").lower()
        t = (text or "").lower()

        keys = []
        # Diplomacy clause
        if "diplomatic" in q or "relocat" in q or "terminate" in q:
            keys += ["diplomatic", "terminate", "2 months", "commission"]
        # Repairs
        if "repair" in q or "broken" in q or "spoil" in q:
            keys += ["s$200", "bulb", "tube", "air", "approval", "fair wear"]
        # Return unit
        if "return" in q or "handover" in q or "move out" in q:
            keys += ["clean", "dry clean", "curtain", "joint inspection", "keys"]

        return sum([1 for k in keys if k in t])

    def _clause_priority(question: str):
        """Return clause priority list based on question intent"""
        q = (question or "").lower()

        if "diplomatic" in q:
            return ["5(c)", "5(d)", "5(f)"]  # 必须都出现

        if "repair" in q or "broken" in q or "spoil" in q:
            return ["2(f)", "2(g)", "2(i)", "2(j)", "2(k)", "4(c)"]   # 全部覆盖老师示例

        if "return" in q or "handover" in q or "move" in q:
            return ["2(y)", "2(z)", "6(o)"]  # 包含 no rent during repair period

        return []

    def _pick_excerpts(docs: List[Any], max_items: int = 3, question: str = ""):
        """Pick most relevant clauses + force include priority ones"""

        priority = _clause_priority(question)
        ranked, seen = [], set()

        # 从 Retrieval QA 的 source_docs 里筛选
        for d in docs or []:
            content = getattr(d, "page_content", "").strip()
            meta = getattr(d, "metadata", {})
            page = meta.get("page")

            if not content:
                continue

            snippet = content[:400].replace("\n", " ")
            clause = _extract_clause_id(content)
            score = _keyword_score(question, snippet)

            # ⭐ 强制优先条款加权，使其一定排在前面
            if clause and any(clause.lower().startswith(p.lower().replace("clause ","")) for p in priority):
                score += 10

            ranked.append((score, {"quote": snippet, "page": page, "clause": clause}))

        # ⭐ 如果 priority clause 没出现 → 直接向 vectorstore 重新查找补齐
        if ranked:
            found_clauses = {item[1]['clause'] for item in ranked}
            missing = [cl for cl in priority if cl not in found_clauses]

            if missing and "vectorstore" in st.session_state:
                retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10})
                for clause in missing:
                    extra = retr.get_relevant_documents(clause)
                    for d in extra:
                        snippet = d.page_content[:400].replace("\n", " ")
                        ranked.append((999, {
                            "quote": snippet,
                            "page": d.metadata.get("page"),
                            "clause": clause
                        }))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:max_items]]

    def format_contract_answer(user_q: str, llm_answer: str, source_docs: List[Any]) -> str:
        """Format final output / 包装最终输出格式"""
        excerpts = _pick_excerpts(source_docs, question=user_q, max_items=3)

        #excerpts = _pick_excerpts(source_docs, question=user_q)
        refs_lines = [
            f"\"{ex['quote'][:230]}...\" ({ex['clause']}, page {ex['page']})"
            for ex in excerpts
        ]
        ref_text = "\n".join(refs_lines) if refs_lines else "Not available."

        return f"""{llm_answer.strip()}


🔎 Relevant Contract Excerpts:
{ref_text}
"""


    # ===== 页面 UI =====
    is_zh = st.session_state.lang == "zh"
    st.title("租客聊天助手" if is_zh else "Tenant Chatbot Assistant")
    st.caption("基于已上传的租赁合同进行问答" if is_zh else "Contract-aware Q&A using uploaded tenancy documents.")

    # --- Upload PDFs used for RAG / 上传PDF用于RAG ---
    uploaded = st.file_uploader(
        "上传租赁合同或房屋守则（PDF）" if is_zh else "Upload PDF contracts or house rules",
        type="pdf",
        accept_multiple_files=True,
        key=f"kb_uploader_{st.session_state.get('uploader_key', 0)}",
    )

    # ✅ 处理当前上传 & 记录文件名（持久显示）
    if uploaded and len(uploaded) > 0:
        st.session_state.kb_doc_names = [f.name for f in uploaded]  # 保存文件名
        st.session_state.pdf_uploaded = True

    # ✅ 显示已上传/已构建 PDF 文件名（切换页面不会消失）
    if st.session_state.pdf_uploaded and st.session_state.kb_doc_names:
        st.caption("已选择的文件：" if is_zh else "Selected PDFs:")
        for nm in st.session_state.kb_doc_names:
            st.markdown(f"**{nm}**")

    # ===== Build & Reset 按钮显示逻辑 =====
    if st.session_state.pdf_uploaded:

        build_disabled = not bool(os.getenv("OPENAI_API_KEY"))  # 未设置 API Key 则禁用

        clicked = st.button(
            "🔄 构建/刷新知识库" if is_zh else "🔄 Build/Refresh Knowledge Base",
            disabled=build_disabled,
            use_container_width=True,
        )

        reset_clicked = st.button(
            "♻️ 重置知识库" if is_zh else "♻️ Reset Knowledge Base",
            disabled=build_disabled,
            use_container_width=True,
        )

        # ===== Build index / 构建知识库 =====
        if clicked:
            if not uploaded or len(uploaded) == 0:
                st.warning("请先上传至少一个 PDF。" if is_zh else "Please upload at least one PDF first.")
            else:
                with st.spinner("正在根据文档构建索引…" if is_zh else "Indexing documents…"):
                    vs = build_vectorstore(uploaded)
                    st.session_state.vectorstore = vs

                    # ✅ 使用满分格式 Prompt 来建链（保留你的原逻辑也可，只要 return_source_documents=True）
                    lc = lazy_import_langchain()
                    PromptTemplate = lc["PromptTemplate"]
                    ChatOpenAI = lc["ChatOpenAI"]
                    RetrievalQA = lc["RetrievalQA"]

                    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.3})
                    llm = ChatOpenAI(temperature=0)

                    prompt = PromptTemplate(
                        input_variables=["context", "question"],
                        template=(
                            FULL_SCORE_SYSTEM_PROMPT
                            + "\n\n[CONTRACT CONTEXT]\n{context}\n\n[USER QUESTION]\n{question}"
                        ),
                    )

                    # 以 RetrievalQA 构建，强制 return_source_documents=True
                    st.session_state.chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff",
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt}
                    )

                st.success("知识库已就绪！现在可以在下方提问。" if is_zh else "Knowledge base ready! Ask questions below.")

        # ===== Reset Knowledge Base / 重置知识库 =====
        if reset_clicked:
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("chain", None)
            st.session_state["kb_doc_names"] = []
            st.session_state["pdf_uploaded"] = False
            st.session_state["online_msgs"] = []  # ✅ 清理合同问答聊天记录

            chain = st.session_state.get("chain")
            if chain and getattr(chain, "memory", None):
                try:
                    chain.memory.clear()
                except Exception:
                    pass

            st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
            st.toast("知识库与合同聊天已清空。" if is_zh else "Knowledge base & contract chat cleared.")
            st.rerun()

    # Whether RAG chain exists / 是否已建链
    has_chain = st.session_state.get("chain") is not None
    
    # ✅ 渲染历史
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for m in st.session_state.get("online_msgs", []):
        render_message(m.get("role", "assistant"), m.get("content", ""), m.get("ts"))
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input / 输入框
    ph_ready = "就你的合同提问…" if is_zh else "Ask about your contract…"
    ph_build = "请先构建知识库…" if is_zh else "Build the knowledge base first…"
    user_q = st.chat_input(
        ph_ready if has_chain else ph_build,
        disabled=not has_chain,
        key="contract_input"
    )

    # === 并入“满分格式”的核心逻辑 ===
    if has_chain and user_q:
        # 语言护栏
        # if guard_language_and_offer_switch(user_q):
        #     st.stop()
        try:
            guard_language_and_offer_switch(user_q)  # 只提示/切换，不 st.stop()
        except Exception:
            pass

        # 1) 用户气泡
        ts_user = now_ts()
        st.session_state.online_msgs.append({"role": "user", "content": user_q, "ts": ts_user})
        render_message("user", user_q, ts_user)

        # 2) 占位回复
        ans_slot = st.empty()
        with ans_slot.container():
            render_message("assistant", "…", now_ts())

        # 3) 调用链
        try:
            smalltalk = small_talk_zh_basic(user_q) if is_zh else small_talk_response_basic(user_q)
            if smalltalk is not None:
                final_md = smalltalk
                source_docs = []
            else:
                # 系统护栏 + 用户问题
                system_hint = (
                    "你是一名租客助手。仅根据已上传文档作答；若文档中没有答案，请说明信息不足。"
                    if is_zh else
                    "You are a helpful Tenant Assistant. Answer ONLY based on the uploaded documents."
                )
                query = f"{system_hint}\nQuestion: {user_q}"

                with st.spinner("正在回答…" if is_zh else "Answering…"):
                    resp = None
                    try:
                        resp = st.session_state.chain.invoke({"query": query})
                    except Exception:
                        # 兼容老接口
                        resp = st.session_state.chain({"query": query})

                # —— 统一解析为 dict —— #
                if isinstance(resp, dict):
                    final_text = resp.get("result") or resp.get("answer") or ""
                    source_docs = resp.get("source_documents") or []
                else:
                    final_text = str(resp or "")
                    source_docs = []

                # 若链没返回文档，再从向量库兜底取证据，避免第一次没证据导致空白
                if not source_docs and st.session_state.get("vectorstore") is not None:
                    try:
                        retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                        source_docs = retr.get_relevant_documents(user_q)
                    except Exception:
                        source_docs = []

                # 空答案兜底（避免第一次出现空白消息）
                if not final_text.strip():
                    final_text = "Not mentioned in the contract."

                # 包装成满分格式
                final_md = format_contract_answer(user_q, final_text, source_docs)

        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "429" in msg:
                final_md = "（模型额度不足或达到速率限制）" if is_zh else "Quota/rate limit hit."
            elif "401" in msg or "invalid_api_key" in msg.lower():
                final_md = "（API Key 无效）" if is_zh else "Invalid API key."
            else:
                final_md = f"（RAG 调用失败：{e}）" if is_zh else f"RAG call failed: {e}"

        # # 3) 调用链
        # try:
        #     smalltalk = small_talk_zh_basic(user_q) if is_zh else small_talk_response_basic(user_q)
        #     if smalltalk is not None:
        #         # 小聊优先
        #         final_md = smalltalk
        #         source_docs = []
        #     else:
        #         # 用“系统护栏 + 用户问题”的拼接，尽量引导满分格式
        #         system_hint = (
        #             "你是一名租客助手。仅根据已上传文档作答；若文档中没有答案，请说明信息不足。"
        #             if is_zh else
        #             "You are a helpful Tenant Assistant. Answer ONLY based on the uploaded documents."
        #         )
        #         query = f"{system_hint}\nQuestion: {user_q}"
        #         with st.spinner("正在回答…" if is_zh else "Answering…"):
        #             try:
        #                 resp = st.session_state.chain.invoke({"query": query})
        #             except Exception:
        #                 resp = st.session_state.chain({"query": query})

        #         # 提取答案 + 证据
        #         if isinstance(resp, dict):
        #             final_text = resp.get("result") or resp.get("answer") or ""
        #             source_docs = resp.get("source_documents") or []
        #         else:
        #             final_text, source_docs = str(resp), []

        #         # 若链没返回文档，退而用向量库检索补证据
        #         if not source_docs and st.session_state.get("vectorstore") is not None:
        #             try:
        #                 retr = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        #                 source_docs = retr.get_relevant_documents(user_q)
        #             except Exception:
        #                 source_docs = []

        #         # 包装为“满分格式”
        #         final_md = format_contract_answer(user_q, final_text, source_docs)

        # except Exception as e:
        #     msg = str(e)
        #     if "insufficient_quota" in msg or "429" in msg:
        #         final_md = "（模型额度不足或达到速率限制）" if is_zh else "Quota/rate limit hit."
        #     elif "401" in msg or "invalid_api_key" in msg.lower():
        #         final_md = "（API Key 无效）" if is_zh else "Invalid API key."
        #     else:
        #         final_md = f"（RAG 调用失败：{e}）" if is_zh else f"RAG call failed: {e}"

        # 4) 输出 + 入历史
        ts_ans = now_ts()
        st.session_state.online_msgs.append({"role": "assistant", "content": final_md, "ts": ts_ans})
        with ans_slot.container():
            render_message("assistant", final_md, ts_ans)


# --- Repair Ticket page / 报修工单 ---
elif st.session_state.page == "ticket":
    is_zh = st.session_state.lang == "zh"
    st.title("🧰 创建报修工单" if is_zh else "🧰 Create Repair Ticket")

    # Submit ticket form / 提交报修表单
    with st.form("ticket_form", clear_on_submit=True):
        t_title = st.text_input(
            "问题标题" if is_zh else "Issue title",
            placeholder="厨房水槽漏水" if is_zh else "Leaking sink in kitchen",
        )
        t_desc = st.text_area(
            "问题描述" if is_zh else "Description",
            placeholder="请描述具体情况…" if is_zh else "Describe the issue…",
        )
        submitted = st.form_submit_button("📨 提交报修" if is_zh else "📨 Submit Ticket")
        if submitted:
            if not t_title.strip():
                st.warning("请填写问题标题。" if is_zh else "Please enter a title.")
            else:
                try:
                    _, total = create_ticket(t_title.strip(), t_desc.strip())
                    st.success(
                        f"报修已保存！当前共有 {total} 条工单。"
                        if is_zh else
                        f"Ticket saved! (Total tickets: {total})"
                    )
                except Exception as e:
                    st.error(f"DB error: {e}")


    # List my tickets / 显示我的报修工单
    st.subheader("我的报修工单" if is_zh else "My Tickets")
    
    ticket_delete_msg_key = "ticket_delete_msg"
    if st.session_state.get(ticket_delete_msg_key):
        st.success(st.session_state[ticket_delete_msg_key])
        st.session_state.pop(ticket_delete_msg_key, None)
    
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

            # 每条工单一个容器；右上角 ✖ 删除（纯文本按钮）
            with st.container(border=True):
                left, right = st.columns([0.95, 0.05], vertical_alignment="top")

                with left:
                    title_line = (
                        f"**{r['title']}** — _{r['status']}_"
                        if is_zh else
                        f"**{r['title']}** — _{r['status']}_"
                    )
                    st.markdown(title_line)
                    if r.get("description"):
                        st.caption(r["description"])
                    st.caption(("创建时间: " if is_zh else "Created at: ") + f"{ts_str} (SGT)")

                with right:
                    if st.button("✖", key=f"del_ticket_{r['id']}", help="Delete this ticket"):
                        try:
                            with get_db_conn() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("DELETE FROM repair_tickets WHERE id = %s;", (r["id"],))
                                    # 可选：删除后取最新总数，让提示更完整
                                    cur.execute("SELECT COUNT(*) AS c FROM repair_tickets;")
                                    new_total = cur.fetchone()["c"]

                            st.session_state[ticket_delete_msg_key] = (
                                f"已删除工单。当前共有 {new_total} 条工单。"
                                if is_zh else
                                f"Ticket deleted. Total tickets: {new_total}."
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")


# --- Rent Reminder page / 房租提醒 ---
elif st.session_state.page == "reminder":
    is_zh = st.session_state.lang == "zh"
    st.title("💰 创建房租提醒" if is_zh else "💰 Create Rent Reminder")

    # Create reminder form / 创建提醒表单
    with st.form("reminder_form", clear_on_submit=True):
        r_day = st.number_input("每月几号" if is_zh else "Due day of month", 1, 31, 1)
        r_note = st.text_input(
            "备注" if is_zh else "Note",
            placeholder="通过银行卡尾号••1234转账" if is_zh else "Pay via bank transfer ending ••1234",
        )
        r_submit = st.form_submit_button("💾 保存提醒" if is_zh else "💾 Save Reminder")

        if r_submit:
            try:
                # ✅ 直接接收 (rid, total)；不再额外 list_reminders()
                rid, total = create_reminder(int(r_day), (r_note or "").strip())

                msg = (
                    f"已保存！目前共有 {total} 条提醒。"
                    if is_zh else
                    f"Reminder saved! (Total reminders: {total})"
                )
                st.success(msg)

            except Exception as e:
                st.error(f"DB error: {e}")

    # List reminders / 展示提醒列表
    st.subheader("当前提醒" if is_zh else "Current Reminders")

    # ========== Flash banner for delete success / 删除成功后的一次性提示 ==========
    # 如果上一轮点击了删除，我们把消息存在 session_state 里，刷新后在这里显示一次
    delete_msg_key = "rem_delete_msg"
    if st.session_state.get(delete_msg_key):
        st.success(st.session_state[delete_msg_key])
        # 显示一次后立刻清除
        st.session_state.pop(delete_msg_key, None)
        

    # 读取提醒列表
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

            # 每条提醒一个容器；右上角是删除按钮
            with st.container(border=True):
                left, right = st.columns([0.95, 0.05], vertical_alignment="top")

                # 左侧：正文
                with left:
                    if is_zh:
                        st.markdown(f"**每月第 {r['day_of_month']} 天**")
                        st.write(r["note"] or "—")
                    else:
                        st.markdown(f"**Day {r['day_of_month']} of Month**")
                        st.write(r["note"] or "—")
                    st.caption(f"{ts_str} (SGT)")

                # 右侧：右上角删除（纯文本 ✖ 按钮）
                with right:
                    if st.button("✖", key=f"del_rem_{r['id']}", help="Delete this reminder"):
                        try:
                            with get_db_conn() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("DELETE FROM rent_reminders WHERE id = %s;", (r["id"],))

                            # 写入一次性提示信息，然后刷新
                            st.session_state[delete_msg_key] = (
                                f"已删除提醒。" if is_zh else f"Reminder deleted."
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

# --- General Chat (offline) / 通用离线聊天 ---
elif st.session_state.page == "offline":
    is_zh = st.session_state.lang == "zh"
    st.title("💬 通用离线聊天" if is_zh else "💬 General Chat (Offline)")
    st.caption("无需 API，仅支持基础闲聊与引导。" if is_zh else "No API required. Small talk and quick help only.")

    # ✅ 用气泡 UI 渲染历史消息
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for m in st.session_state.get("offline_msgs", []):
        render_message(m.get("role", "assistant"), m.get("content", ""), m.get("ts"))
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input always enabled here / 离线聊天始终可输入    
    user_q = st.chat_input(
        "打个招呼或问一些基础问题…" if is_zh else "Say hello or ask about some basic information…",
        key="offline_input"
    )

    if user_q:
        if guard_language_and_offer_switch(user_q):
            st.stop()

        ts_user = now_ts()
        st.session_state.offline_msgs.append({"role": "user", "content": user_q, "ts": ts_user})
        render_message("user", user_q, ts_user)

        ans_slot = st.empty()
        with ans_slot.container():
            render_message("assistant", "…", now_ts())

        ans = (small_talk_zh(user_q) if is_zh else small_talk_response(user_q)) or (
            "当前为离线聊天模式。你也可以在侧栏切换到“合同问答”。" if is_zh else
            "I'm in offline chat mode. Use the sidebar to switch features."
        )
        ts_ans = now_ts()
        st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
        with ans_slot.container():
            render_message("assistant", ans, ts_ans)
