#!/usr/bin/env python3
"""
Tenant Chatbot Assistant (Streamlit + RAG)
Simplified version with sidebar navigation buttons to switch between tenant utilities.

Run:
  streamlit run app_tenantbot.py
"""
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

import os
import tempfile
import re
from datetime import datetime
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

st.set_page_config(
    page_title="Tenant Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.sidebar-btn {width:100%; text-align:left; background:#e3f2fd; border:none; padding:0.5rem 1rem; border-radius:0.5rem; margin:0.2rem 0;}
.sidebar-btn:hover {background:#bbdefb;}
</style>
""", unsafe_allow_html=True)

# ---------------------- Session Init ----------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"   # "en" or "zh"

# --- initialize session state ---
if "page" not in st.session_state:
    st.session_state.page = "offline"   # ✅ 默认进入离线聊天页面
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
        index=0 if st.session_state.get("lang", "en") == "en" else 1
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
        caption_text = "Upload PDFs anytime. Enable the build button by setting OPENAI_API_KEY in the expander above or via .env."
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
    
    # ====== Clear Chat ======
    if st.button(clear_label, use_container_width=True):
        st.session_state.setdefault("offline_msgs", [])
        st.session_state.setdefault("online_msgs", [])
        st.session_state.offline_msgs.clear()
        st.session_state.online_msgs.clear()

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
    # Build embeddings using environment-provided OPENAI_API_KEY
    paths = []
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

    embeddings = OpenAIEmbeddings()  # reads OPENAI_API_KEY from env/.env
    vs = FAISS.from_documents(texts, embeddings)
    for p in paths:
        try:
            os.unlink(p)
        except Exception:
            pass
    return vs

def create_chain(vs):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

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
    return chain



# ---------------------- Utilities ----------------------
def now_ts(lang: str) -> str:
    """Timestamp formatted by language."""
    if lang == "zh":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===== Small-talk helpers (shared) =====
def normalize_word(word: str) -> str:
    word = word.lower()
    suffixes = ["ing","ed","es","s","ly","tion","ions","ness","ment","ments","ities","ity","als","al","ers","er"]
    for suf in suffixes:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[:-len(suf)]
    return word

def normalize_text(text: str) -> str:
    words = re.findall(r"[a-zA-Z\u4e00-\u9fff']+", text.lower())
    return " ".join(normalize_word(w) for w in words)

def normalize_text_zh(text: str) -> str:
    """Keep Chinese and ASCII letters/numbers for ZH pipeline."""
    # 兼容中英混输，但中文匹配主要靠包含判断
    return "".join(re.findall(r"[0-9A-Za-z\u4e00-\u9fff'，。！？、：；（）()《》“”\"' ]+", text))


def any_terms_en(text_norm: str, terms: list[str]) -> bool:
    """Exact-ish word match for EN."""
    for t in terms:
        t2 = normalize_word(t)
        if re.search(rf"\b{re.escape(t2)}\b", text_norm) or t2 in text_norm:
            return True
    return False

def contains_any_zh(text_norm: str, phrases: list[str]) -> bool:
    """Substring contains for ZH."""
    return any(p in text_norm for p in phrases)


def any_phrases(text: str, phrases: list[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(p) in norm for p in phrases)

# ---------------------- Small-talk (ZH) ----------------------
def small_talk_zh(q_raw: str) -> str | None:
    q = normalize_text_zh(q_raw.strip())
    # 问候
    if contains_any_zh(q, ["你好","您好","嗨","哈喽","早上好","下午好","晚上好"]):
        return "你好！我是你的租客小助手 👋 有什么可以帮你的？"
    # 近况
    if contains_any_zh(q, ["你好吗","最近怎么样","最近如何","最近还好么"]):
        return "我很好，随时待命～你有什么想了解的？"
    # 身份
    if contains_any_zh(q, ["你是谁","你是干什么的","你叫什么名字"]):
        return "我是帮助租客进行简单咨询的聊天助手（离线模式）。"
    # 感谢
    if contains_any_zh(q, ["谢谢","多谢","非常感谢","感谢你","太感谢了"]):
        return "不客气～还有什么我能帮忙的吗？"
    # 能力
    if contains_any_zh(q, ["能做什么","会干嘛","你能帮我什么","可以做什么"]):
        return "我可以进行问候与基础问答，并指引你创建报修或设置租金提醒。此离线版不支持合同问答。"
    # 指引
    if contains_any_zh(q, ["怎么开始","如何使用","怎么用","使用说明"]):
        return "你可以在侧栏切换语言进行或清空聊天记录。也可以问我打招呼、功能说明等基础问题。"
    # 租金提醒
    if contains_any_zh(q, ["租金提醒","房租提醒","什么时候交房租","交租提醒"]):
        return "你可以自己每月记个备忘；完整版本里我可以替你保存提醒。"
    # 报修
    if contains_any_zh(q, ["报修","维修","漏水","坏了","修理","故障"]):
        return "请简单描述问题。完整版本中我可以帮你提交报修给物业。"

    return None

def small_talk_zh_basic(q_raw: str) -> str | None:
    """中文在线小聊：若像是合同/维修等检索类问题，则返回 None，让 RAG 去答；否则返回小聊回复。"""
    q = normalize_text_zh(q_raw.strip())
    # 这些关键词一旦出现，就认为是合同/知识库问题，不走小聊
    contract_like = ["合同","租约","条款","租金","押金","房东","租客","维修","报修","终止","违约","续约","账单","费用"]
    if contains_any_zh(q, contract_like):
        return None
    return small_talk_zh(q_raw)


# ---------------------- Small-talk (EN) ----------------------
# 完整版（离线用）
def small_talk_response(q_raw: str) -> str | None:
    q = normalize_text(q_raw.strip())
    # greetings
    if any_terms_en(q, ["hi","hello","hey","morning","evening","afternoon"]) or any_phrases(q, ["你好","嗨","哈喽"]):
        return "Hello! I’m your Tenant Assistant 👋 How can I help you today?"
    # how are you
    if any_phrases(q, ["how are you","how's it going","how are u","how are ya","how are things","how do you feel","你好吗","最近怎么样","最近如何"]):
        return "I'm doing well and ready to help! How can I assist you today?"
    # who are you
    if any_phrases(q, ["who are you","what are you","your name","你是谁","你是干什么的"]):
        return "I’m a friendly chatbot that helps tenants understand contracts and manage repairs or rent reminders."
    # thanks
    if any_terms_en(q, ["thanks","thank","thx","appreciate"]) or any_phrases(q, ["thank you","many thanks","谢谢","多谢","非常感谢","感謝"]):
        return "You're welcome! If there’s anything else you need, just let me know."
    # capabilities
    if any_phrases(q, ["what can you do","what can u do","能做什么","你会干嘛"]) or any_terms_en(q, ["function","feature","capability"]):
        return ("I can help you read tenancy agreements, create repair tickets, and set rent reminders. "
                "Once you add an API key, I can also answer contract questions directly!")
    # upload/how to start
    if any_phrases(q, ["how to upload","upload pdf","add document","how to start","start upload","怎么上传","如何开始"]):
        return ("Click **‘Upload PDF contracts or house rules’** to add documents. "
                "Then click **‘Build/Refresh Knowledge Base’** after setting your API key.")
    # rent reminder
    if any_phrases(q, ["rent reminder","rent day","when to pay rent","租金提醒","什么时候交房租"]):
        return "Use **💰 Create Rent Reminder** in the sidebar to set a monthly reminder."
    # repair
    if any_terms_en(q, ["repair","maintain","fix","broken","leak","damage","fault","issue"]) or any_phrases(q, ["报修","维修","漏水","坏了"]):
        return "Use **🧰 Create Repair Ticket** in the sidebar. Describe the problem and I’ll record it."
    # contract mention
    if any_terms_en(q, ["contract","agreement","lease","term","clause","deposit","renewal","policy","rules"]) or any_phrases(q, ["合同","条款","押金","续约","租约"]):
        return "Upload your contract and set an API key; I’ll then answer questions based on the document."
    return None

# 基础版（在线用：不拦截合同问题）
def small_talk_response_basic(q_raw: str) -> str | None:
    q = normalize_text(q_raw.strip())
    
    if any_terms_en(q, [
        "contract", "agreement", "lease", "tenant", "landlord", "deposit", "repair",
        "maintenance", "damage", "clause", "policy", "rent", "renewal", "notice", "terminate"
    ]):
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
    # ===== end helpers =====

# --- page: contract chat ---
if st.session_state.page == "chat":
    lang = st.session_state.get("lang", "en")

    # ---------------- 文案字典 ----------------
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
        api_hint = "API key detected. Upload PDFs and click **Build/Refresh Knowledge Base** to enable contract Q&A. You can still have a quick small talk below."
        chat_ph_offline = "Say hello or ask about some basic information…"
        chat_ph_online = "Ask about your contract…"
        dep_missing = "LangChain not installed. Run: pip install langchain langchain-openai openai pypdf faiss-cpu"
        idx_spinner = "Indexing documents…"
        idx_done = "Knowledge base ready! Ask questions below."
        ans_spinner = "Answering…"
        offline_hint = ("I'm in offline chat mode. You can explore the sidebar features, "
                        "or switch to Contract Chat for document-based Q&A.")

    # ---------------- 页面头部 ----------------
    st.title(f"🤖 {title}")
    st.caption(subtitle)

    if not LANGCHAIN_AVAILABLE:
        st.error(dep_missing)
        st.stop()

    # ---------------- 上传与构建 ----------------
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

    # ---------------- 历史状态 ----------------
    st.session_state.setdefault("offline_msgs", [])
    st.session_state.setdefault("online_msgs", [])

    # ---------------- 离线模式（无 API Key） ----------------
    if not os.getenv("OPENAI_API_KEY"):
        st.info(offline_banner)
        for m in st.session_state.offline_msgs:
            with st.chat_message(m["role"]):
                if m.get("ts"): st.caption(m["ts"])
                st.markdown(m["content"])

        user_q = st.chat_input(chat_ph_offline)
        if user_q:
            ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.offline_msgs.append({"role": "user", "content": user_q, "ts": ts_now})
            with st.chat_message("user"):
                st.caption(ts_now); st.markdown(user_q)

            ans = (
                small_talk_zh(user_q) if lang == "zh" else small_talk_response(user_q)
            ) or offline_hint

            ts_ans = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
            with st.chat_message("assistant"):
                st.caption(ts_ans); st.markdown(ans)
        st.stop()

    # ---------------- 有 API 但未构建索引（暂时闲聊） ----------------
    if os.getenv("OPENAI_API_KEY") and "chain" not in st.session_state:
        st.info(api_hint)
        for m in st.session_state.offline_msgs:
            with st.chat_message(m["role"]):
                if m.get("ts"): st.caption(m["ts"])
                st.markdown(m["content"])

        tmp_q = st.chat_input(chat_ph_offline)
        if tmp_q:
            ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.offline_msgs.append({"role": "user", "content": tmp_q, "ts": ts_now})
            with st.chat_message("user"):
                st.caption(ts_now); st.markdown(tmp_q)

            ans = (
                small_talk_zh(tmp_q) if lang == "zh" else small_talk_response(tmp_q)
            ) or build_help_off
            ts_ans = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
            with st.chat_message("assistant"):
                st.caption(ts_ans); st.markdown(ans)
        st.stop()
                
    # ---------------- 在线 RAG 问答模式 ----------------
    if "chain" in st.session_state:
        for m in st.session_state.online_msgs:
            with st.chat_message(m["role"]):
                if m.get("ts"): st.caption(m["ts"])
                st.markdown(m["content"])

        user_q = st.chat_input(chat_ph_online)
        if user_q:
            ts_user = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.online_msgs.append({"role": "user", "content": user_q, "ts": ts_user})
            with st.chat_message("user"):
                st.caption(ts_user); st.markdown(user_q)

            # ✅ 先走小闲聊：命中则不调用向量链
            if lang == "zh":
                smalltalk = small_talk_zh_basic(user_q)
            else:
                smalltalk = small_talk_response_basic(user_q)

            if smalltalk is not None:
                final_md = smalltalk  # 你也可以加个前缀，比如 "[offline] "
            else:
                with st.spinner(ans_spinner):
                    system_hint = (
                        "You are a helpful Tenant Assistant. Answer ONLY based on the uploaded documents. "
                        "If the answer isn't present in the documents, say you don't have enough information."
                        if lang == "en"
                        else "你是一名租客助手。仅根据已上传文档作答；若文档中没有答案，请说明信息不足。"
                    )
                    query = f"{system_hint}\nQuestion: {user_q}"
                    resp = st.session_state.chain.invoke({"question": query})
                    final_md = resp.get("answer", "（暂无答案）" if lang == "zh" else "(no answer)")

            ts_ans = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.online_msgs.append({"role": "assistant", "content": final_md, "ts": ts_ans})
            with st.chat_message("assistant"):
                st.caption(ts_ans); st.markdown(final_md)

# --- page: repair ticket ---
elif st.session_state.page == "ticket":
    if st.session_state.get("lang", "en") == "zh":
        st.title("🧰 创建报修工单")
        issue_label = "问题标题"
        issue_ph = "厨房水槽漏水"
        desc_label = "问题描述"
        desc_ph = "请描述具体情况…"
        submit_btn = "📨 提交报修"
        created_ok = "报修已创建！"
        my_tickets = "我的报修工单"
        status_open = "进行中"
    else:
        st.title("🧰 Create Repair Ticket")
        issue_label = "Issue title"
        issue_ph = "Leaking sink in kitchen"
        desc_label = "Description"
        desc_ph = "Describe the issue…"
        submit_btn = "📨 Submit Ticket"
        created_ok = "Ticket created!"
        my_tickets = "My Tickets"
        status_open = "open"

    with st.form("ticket_form", clear_on_submit=True):
        t_title = st.text_input(issue_label, placeholder=issue_ph)
        t_desc = st.text_area(desc_label, placeholder=desc_ph)
        submitted = st.form_submit_button(submit_btn)
        if submitted and t_title:
            st.session_state.tickets.append({"title": t_title, "desc": t_desc, "status": status_open})
            st.success(created_ok)

    if st.session_state.tickets:
        st.subheader(my_tickets)
        for i, tk in enumerate(st.session_state.tickets):
            st.markdown(f"**{i+1}. {tk['title']}** – _{tk['status']}_")
            st.caption(tk['desc'])

# --- page: rent reminder ---
elif st.session_state.page == "reminder":
    if st.session_state.get("lang", "en") == "zh":
        st.title("💰 创建房租提醒")
        day_label = "每月几号"
        note_label = "备注"
        note_ph = "通过银行卡尾号••1234转账"
        save_btn = "💾 保存提醒"
        saved_ok = "提醒已保存！"
        current_title = "当前提醒"
        every_month_on = "每月的第 **{day}** 天 — {note}"
    else:
        st.title("💰 Create Rent Reminder")
        day_label = "Due day of month"
        note_label = "Note"
        note_ph = "Pay via bank transfer ending ••1234"
        save_btn = "💾 Save Reminder"
        saved_ok = "Reminder saved!"
        current_title = "Current Reminder"
        every_month_on = "Every month on day **{day}** — {note}"

    with st.form("reminder_form", clear_on_submit=True):
        r_day = st.number_input(day_label, 1, 31, 1)
        r_note = st.text_input(note_label, placeholder=note_ph)
        r_submit = st.form_submit_button(save_btn)
        if r_submit:
            st.session_state.reminders = [{"day": int(r_day), "note": r_note}]
            st.success(saved_ok)

    if st.session_state.reminders:
        st.subheader(current_title)
        for r in st.session_state.reminders:
            st.write(every_month_on.format(day=r["day"], note=r["note"]))

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
        offline_hint = ("I'm in offline chat mode. You can explore the sidebar features, "
                        "or switch to Contract Chat for document-based Q&A.")

    if "offline_msgs" not in st.session_state:
        st.session_state.offline_msgs = []

    # 渲染历史
    for m in st.session_state.offline_msgs:
        with st.chat_message(m["role"]):
            if m.get("ts"): st.caption(m["ts"])
            st.markdown(m["content"])

    # 输入
    user_q = st.chat_input(history_empty_hint)
    if user_q:
        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.offline_msgs.append({"role": "user", "content": user_q, "ts": ts_now})
        with st.chat_message("user"):
            st.caption(ts_now); st.markdown(user_q)

        # 按语言分别走触发词路由（你已实现 small_talk_en / small_talk_zh）
        if lang == "zh":
            ans = (small_talk_zh(user_q) or offline_hint)
        else:
            ans = (small_talk_response(user_q) or offline_hint)

        ts_ans = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.offline_msgs.append({"role": "assistant", "content": ans, "ts": ts_ans})
        with st.chat_message("assistant"):
            st.caption(ts_ans); st.markdown(ans)