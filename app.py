"""
AI Video Assistant — Streamlit UI
Wraps main.run_pipeline() with a polished chat + insights interface.
"""

import streamlit as st
import tempfile
import os
from datetime import datetime

from main import run_pipeline
from core.rag_engine import ask_question

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #1a1f3a 0%, #0e1117 45%, #0a0c10 100%);
    }

    /* Hero header */
    .hero {
        padding: 2.2rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.35);
        margin-bottom: 1.6rem;
    }
    .hero h1 {
        color: white; font-weight: 800; font-size: 2.1rem; margin: 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.9); font-size: 1.02rem; margin-top: 0.4rem;
    }

    /* Feature pill cards in sidebar */
    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .feature-card b { color: #c4b5fd; }
    .feature-card span { color: rgba(255,255,255,0.65); font-size: 0.86rem; }

    /* Metric-style badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        color: #c4b5fd;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    /* Content cards */
    .content-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
    }

    div[data-testid="stChatInput"] textarea { border-radius: 12px !important; }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        color: white;
        padding: 0.6rem 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #7c7ff0, #a879f7);
        color: white;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.1rem;
        background: rgba(255,255,255,0.03);
    }

    /* Lock the sidebar permanently open — hide the collapse control entirely */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        min-width: 320px !important;
        max-width: 320px !important;
        width: 320px !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing_log" not in st.session_state:
    st.session_state.processing_log = []

# --------------------------------------------------------------------------
# Sidebar — input + feature explainer
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 AI Video Assistant")
    st.caption("Turn any meeting or video into structured insight — then chat with it.")

    st.markdown("#### ⚙️ Source")
    input_mode = st.radio(
        "Input type",
        ["📤 Upload from device", "🔗 YouTube URL", "📁 Local file path"],
        label_visibility="collapsed",
    )

    source = None
    uploaded_file = None

    if input_mode == "📤 Upload from device":
        uploaded_file = st.file_uploader(
            "Upload a video or audio file",
            type=["mp4", "mov", "mkv", "avi", "webm", "mp3", "wav", "m4a"],
            help="Max size depends on your Streamlit server config (default 200MB).",
        )
        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"📎 {uploaded_file.name} · {size_mb:.1f} MB")
    elif input_mode == "🔗 YouTube URL":
        source = st.text_input("Paste a YouTube link", placeholder="https://youtube.com/watch?v=...")
    else:
        source = st.text_input("Local file path", placeholder="/path/to/video.mp4")

    run_clicked = st.button("🚀 Process Video", use_container_width=True)

    st.divider()
    st.markdown("#### ✨ What this does")
    features = [
        ("📤 Device Upload", "Drop in a video/audio file straight from your computer."),
        ("🎧 Audio Extraction", "Pulls & chunks audio from a YouTube link or local file."),
        ("📝 Transcription", "Converts speech to text across all chunks."),
        ("🏷️ Auto Title", "Generates a concise title from the transcript."),
        ("📋 Summary", "Condenses the full discussion into key points."),
        ("✅ Action Items", "Extracts concrete follow-up tasks."),
        ("🔑 Key Decisions", "Surfaces decisions made during the meeting."),
        ("❓ Open Questions", "Flags unresolved questions worth revisiting."),
        ("💬 Chat (RAG)", "Ask free-form questions grounded in the transcript."),
    ]
    for name, desc in features:
        st.markdown(f"""<div class="feature-card"><b>{name}</b><br><span>{desc}</span></div>""",
                     unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 AI Video Assistant</h1>
    <p>Upload a talk, meeting, or video URL — get a title, summary, action items, decisions,
    open questions, and a chat interface grounded in the transcript.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Run pipeline with live status
# --------------------------------------------------------------------------
if run_clicked:
    # Resolve the actual source path/URL depending on chosen input mode
    resolved_source = None
    temp_path = None

    if input_mode == "📤 Upload from device":
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.getbuffer())
            tmp.close()
            temp_path = tmp.name
            resolved_source = temp_path
    else:
        if source and source.strip():
            resolved_source = source.strip()

    if not resolved_source:
        st.error("Please upload a file, or provide a YouTube URL / local file path first.")
    else:
        st.session_state.messages = []
        stages = [
            "Extracting & chunking audio",
            "Transcribing speech to text",
            "Generating title & summary",
            "Extracting action items, decisions & questions",
            "Building chat index (RAG)",
        ]
        progress = st.progress(0, text=stages[0])
        try:
            # We call the real pipeline once; the progress bar is a UX
            # indicator of the stages run_pipeline performs internally.
            progress.progress(15, text=stages[0])
            progress.progress(35, text=stages[1])
            progress.progress(55, text=stages[2])
            progress.progress(75, text=stages[3])
            progress.progress(90, text=stages[4])

            result = run_pipeline(resolved_source)

            progress.progress(100, text="Done!")
            st.session_state.result = result
            display_source = uploaded_file.name if uploaded_file is not None else resolved_source
            st.session_state.processing_log.append(
                {"source": display_source, "time": datetime.now().strftime("%H:%M:%S")}
            )
            st.success(f"✅ Processed: **{result['title']}**")
        except Exception as e:
            progress.empty()
            st.error(f"❌ Pipeline failed: {e}")
            st.exception(e)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
result = st.session_state.result

if result:
    st.markdown(f"## 📌 {result['title']}")
    st.markdown(
        f'<span class="badge">📄 {len(result["transcript"].split())} words</span>'
        f'<span class="badge">✅ {len(result["action_items"]) if isinstance(result["action_items"], list) else "—"} action items</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📝 Transcript", "💬 Chat"]
    )

    def render_list_or_text(content):
        if isinstance(content, (list, tuple)):
            for item in content:
                st.markdown(f"- {item}")
        else:
            st.markdown(content)

    with tab_summary:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(result["summary"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_actions:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        render_list_or_text(result["action_items"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_decisions:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        render_list_or_text(result["key_decisions"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_questions:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        render_list_or_text(result["open_questions"])
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_transcript:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.text_area("Full transcript", result["transcript"], height=400, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_chat:
        st.caption("Ask anything about this video — answers are grounded in the transcript via RAG.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about the video..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], prompt)
                    except Exception as e:
                        answer = f"⚠️ Error answering question: {e}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 👋 Get started")
    st.markdown(
        "Paste a **YouTube URL** or a **local file path** in the sidebar and click "
        "**Process Video** to generate a title, summary, action items, key decisions, "
        "open questions, and an interactive chat over the transcript."
    )
    st.markdown('</div>', unsafe_allow_html=True)