import streamlit as st
import numpy as np
import faiss
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI — YouTube Intelligence BOT",
    page_icon="",
    layout="wide"
)

# ===============================
# CUSTOM CSS — PROFESSIONAL DARK UI
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ============================================
   2026 PREMIUM DARK + RED — YOUTUBE × BOTPRESS
   ============================================ */
:root {
  --yt-red: #FF0033;
  --yt-red-soft: #FF3D5A;
  --yt-red-deep: #B8002A;
  --bg-0: #0A0A0A;
  --bg-1: #121212;
  --bg-2: #1A1A1A;
  --surface: #161616;
  --surface-2: #1E1E1E;
  --border: #2A2A2A;
  --border-soft: #1F1F1F;
  --text: #F5F5F5;
  --text-muted: #9A9A9A;
  --text-dim: #6B6B6B;
  --accent-green: #00D26A;
  --glow-red: 0 0 24px rgba(255, 0, 51, 0.35);
  --glow-red-soft: 0 0 12px rgba(255, 0, 51, 0.18);
  --ease: cubic-bezier(0.4, 0, 0.2, 1);
}

*, *::before, *::after { box-sizing: border-box; }

/* ---------- BASE ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg-0) !important;
  color: var(--text) !important;
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  font-feature-settings: 'cv11', 'ss01';
  -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 800px 500px at 15% -10%, rgba(255, 0, 51, 0.10), transparent 60%),
    radial-gradient(ellipse 700px 400px at 110% 10%, rgba(255, 0, 51, 0.06), transparent 60%),
    radial-gradient(ellipse at 50% 100%, #141414 0%, #0A0A0A 60%) !important;
  background-attachment: fixed !important;
}

[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
  visibility: hidden !important;
}

.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1400px !important;
}

/* ---------- TYPOGRAPHY ---------- */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; font-weight: 700 !important; letter-spacing: -0.02em; }
p, span, div, label { color: var(--text); }
a { color: var(--yt-red-soft) !important; text-decoration: none; }
a:hover { color: var(--yt-red) !important; }

::selection { background: rgba(255, 0, 51, 0.35); color: #fff; }

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0E0E0E 0%, #0A0A0A 100%) !important;
  border-right: 1px solid var(--border-soft) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ---------- HEADER (your .vidmind-header) ---------- */
.vidmind-header {
  display: flex; align-items: center; gap: 16px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.75rem;
  background: rgba(20, 20, 20, 0.6);
  backdrop-filter: blur(20px) saturate(150%);
  -webkit-backdrop-filter: blur(20px) saturate(150%);
  border: 1px solid var(--border);
  border-radius: 18px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
}
.vidmind-logo {
  width: 52px; height: 52px;
  background: linear-gradient(135deg, var(--yt-red) 0%, var(--yt-red-deep) 100%);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px; color: #fff;
  box-shadow: var(--glow-red), inset 0 1px 0 rgba(255,255,255,0.2);
  animation: pulseGlow 3s var(--ease) infinite;
  position: relative;
}
.vidmind-logo::before {
  content: '▶'; color: #fff; font-size: 20px; margin-left: 3px;
}
.vidmind-title {
  font-family: 'Inter', sans-serif !important;
  font-size: 1.85rem; font-weight: 800; line-height: 1;
  background: linear-gradient(90deg, #FFFFFF 0%, #FF4466 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.03em;
}
.vidmind-subtitle {
  font-size: 0.72rem; color: var(--text-dim);
  font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; margin-top: 6px;
}

/* ---------- BADGES ---------- */
.status-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 12px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
}
.status-ready {
  background: rgba(0, 210, 106, 0.10);
  border: 1px solid rgba(0, 210, 106, 0.30);
  color: var(--accent-green) !important;
}
.status-idle {
  background: rgba(154, 154, 154, 0.08);
  border: 1px solid rgba(154, 154, 154, 0.20);
  color: var(--text-muted) !important;
}

/* ---------- CARDS ---------- */
.glass-card {
  position: relative;
  background: linear-gradient(180deg, rgba(30,30,30,0.7), rgba(20,20,20,0.7));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  backdrop-filter: blur(12px);
  transition: transform 0.3s var(--ease), box-shadow 0.3s var(--ease), border-color 0.3s var(--ease);
}
.glass-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255, 0, 51, 0.35);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5), var(--glow-red-soft);
}

/* ---------- METRICS ---------- */
.metric-row { display: flex; gap: 12px; margin: 1rem 0; }
.metric-box {
  flex: 1;
  background: linear-gradient(180deg, #1A1A1A, #141414);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem; text-align: center;
  transition: all 0.25s var(--ease);
}
.metric-box:hover { border-color: var(--yt-red); box-shadow: var(--glow-red-soft); }
.metric-value {
  font-family: 'Inter', sans-serif !important;
  font-size: 1.6rem; font-weight: 800;
  background: linear-gradient(135deg, #FFFFFF, #FF4466);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.metric-label {
  font-size: 0.68rem; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px; font-weight: 600;
}

/* ---------- INPUTS ---------- */
[data-testid="stTextInput"] input,
[data-testid="stChatInput"] textarea {
  background: #141414 !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  padding: 0.85rem 1.1rem !important;
  font-size: 0.95rem !important;
  transition: all 0.25s var(--ease) !important;
  caret-color: var(--yt-red) !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text-dim) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--yt-red) !important;
  box-shadow: 0 0 0 4px rgba(255, 0, 51, 0.12), var(--glow-red-soft) !important;
  outline: none !important;
}
[data-testid="stTextInput"] label { color: var(--text-muted) !important; font-weight: 500 !important; }

/* ---------- BUTTONS ---------- */
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, var(--yt-red) 0%, var(--yt-red-deep) 100%) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.75rem 1.5rem !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  letter-spacing: 0.01em !important;
  cursor: pointer !important;
  transition: all 0.25s var(--ease) !important;
  box-shadow: 0 6px 20px rgba(255, 0, 51, 0.35), inset 0 1px 0 rgba(255,255,255,0.18) !important;
  width: 100% !important;
  position: relative;
  overflow: hidden;
}
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 30px rgba(255, 0, 51, 0.55), inset 0 1px 0 rgba(255,255,255,0.25) !important;
  filter: brightness(1.08);
}
[data-testid="stButton"] button:active { transform: scale(0.97) !important; }

.clear-btn [data-testid="stButton"] button {
  background: rgba(255, 0, 51, 0.08) !important;
  border: 1px solid rgba(255, 0, 51, 0.30) !important;
  color: var(--yt-red-soft) !important;
  box-shadow: none !important;
}
.clear-btn [data-testid="stButton"] button:hover {
  background: rgba(255, 0, 51, 0.18) !important;
  box-shadow: var(--glow-red-soft) !important;
}

/* ---------- TABS ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  padding: 0.85rem 1.5rem !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.25s var(--ease) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: var(--text) !important; }
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--yt-red-soft) !important;
  border-bottom: 2px solid var(--yt-red) !important;
  text-shadow: 0 0 12px rgba(255, 0, 51, 0.4);
}

/* ---------- BOTPRESS-STYLE CHAT ---------- */
[data-testid="stChatMessageContainer"], [data-testid="stVerticalBlock"] [data-testid="stChatMessage"] {
  background: transparent !important;
}
[data-testid="stChatMessage"] {
  background: linear-gradient(180deg, #1A1A1A, #151515) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1rem 1.15rem !important;
  margin-bottom: 0.85rem !important;
  transition: all 0.25s var(--ease);
  animation: slideInLeft 0.4s var(--ease);
}
[data-testid="stChatMessage"]:hover {
  border-color: rgba(255, 0, 51, 0.25);
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: linear-gradient(180deg, rgba(0, 210, 106, 0.06), rgba(0, 210, 106, 0.02)) !important;
  border-color: rgba(0, 210, 106, 0.20) !important;
  animation: slideInRight 0.4s var(--ease);
}
/* Assistant avatar — red */
[data-testid="chatAvatarIcon-assistant"] {
  background: linear-gradient(135deg, var(--yt-red), var(--yt-red-deep)) !important;
  box-shadow: var(--glow-red-soft);
}
[data-testid="chatAvatarIcon-user"] {
  background: linear-gradient(135deg, var(--accent-green), #00A050) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span { color: var(--text) !important; line-height: 1.65; }
[data-testid="stChatMessage"] code {
  background: #0A0A0A !important;
  color: var(--yt-red-soft) !important;
  padding: 2px 6px; border-radius: 6px;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.85em;
  border: 1px solid var(--border);
}
[data-testid="stChatMessage"] pre {
  background: #0A0A0A !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 12px !important;
}

[data-testid="stChatInput"] {
  background: transparent !important;
  border-top: 1px solid var(--border) !important;
  padding-top: 0.75rem !important;
}

/* ---------- EXPANDER ---------- */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 600 !important; }
[data-testid="stExpander"] summary:hover { color: var(--yt-red-soft) !important; }

/* ---------- ALERTS ---------- */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--text) !important;
}

/* ---------- METRIC (st.metric) ---------- */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, #1A1A1A, #141414);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem;
}
[data-testid="stMetricValue"] {
  color: var(--yt-red-soft) !important;
  font-weight: 800 !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* ---------- HELPERS ---------- */
.section-label {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.72rem; font-weight: 700;
  color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.16em;
  margin: 1rem 0 0.6rem 0;
}
.custom-divider {
  height: 1px; margin: 1.5rem 0;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}
.transcript-box {
  background: #0E0E0E;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  font-size: 0.9rem; line-height: 1.75;
  color: #C8C8C8 !important;
  max-height: 320px; overflow-y: auto;
  font-family: 'Inter', sans-serif;
}
.video-info {
  display: flex; align-items: center; gap: 12px;
  background: linear-gradient(135deg, rgba(255,0,51,0.08), rgba(255,0,51,0.02));
  border: 1px solid rgba(255, 0, 51, 0.20);
  border-radius: 12px; padding: 1rem 1.25rem; margin: 1rem 0;
}
.video-url { font-size: 0.85rem; color: var(--yt-red-soft) !important; word-break: break-all; font-family: 'JetBrains Mono', monospace; }
.video-label { font-size: 0.7rem; color: var(--text-dim) !important; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }

/* ---------- SPINNER ---------- */
[data-testid="stSpinner"] > div { border-top-color: var(--yt-red) !important; }

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0A0A0A; }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--yt-red-deep), #4A0010);
  border-radius: 8px;
  border: 2px solid #0A0A0A;
}
::-webkit-scrollbar-thumb:hover { background: var(--yt-red); }

/* ---------- ANIMATIONS ---------- */
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 20px rgba(255,0,51,0.35), inset 0 1px 0 rgba(255,255,255,0.2); }
  50%      { box-shadow: 0 0 35px rgba(255,0,51,0.60), inset 0 1px 0 rgba(255,255,255,0.25); }
}
@keyframes slideInLeft {
  from { opacity: 0; transform: translateX(-12px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
  from { opacity: 0; transform: translateX(12px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 900px) {
  .vidmind-title { font-size: 1.4rem; }
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)
# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embedding_model, groq_client

embedding_model, groq_client = load_models()

# ===============================
# SESSION STATE
# ===============================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks_store" not in st.session_state:
    st.session_state.chunks_store = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "current_url" not in st.session_state:
    st.session_state.current_url = ""
if "word_count" not in st.session_state:
    st.session_state.word_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ===============================
# HELPER FUNCTIONS
# ===============================
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        return url

def get_transcript(url):
    try:
        video_id = extract_video_id(url)
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id)
        return " ".join([item.text for item in transcript_data])
    except Exception as e:
        return f"ERROR: {str(e)}"

def process_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    st.session_state.vector_store = index
    st.session_state.chunks_store = chunks
    st.session_state.chunk_count = len(chunks)

def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = st.session_state.vector_store.search(
        np.array(query_embedding), top_k
    )
    return "\n\n".join([st.session_state.chunks_store[i] for i in indices[0]])

def generate_answer(query):
    context = retrieve_context(query)
    prompt = f"""You are an expert AI assistant that helps users understand YouTube video content.
Use ONLY the context below to answer. Be clear, concise, and insightful.

Context:
{context}

Question:
{query}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="vidmind-header">
    <div class="vidmind-logo"></div>
    <div>
        <div class="vidmind-title">VidMind AI</div>
        <div class="vidmind-subtitle">YouTube Intelligence Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    if st.session_state.vector_store is not None:
        st.markdown('<span class="status-badge status-ready">● Video Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">○ No Video Loaded</span>', unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.session_state.vector_store is not None:
        st.markdown('<div class="section-label">Analytics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-value">{st.session_state.word_count:,}</div>
                <div class="metric-label">Words</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{st.session_state.chunk_count}</div>
                <div class="metric-label">Chunks</div>
            </div>
        </div>
        <div class="metric-box" style="margin-top:0">
            <div class="metric-value">{len(st.session_state.messages)}</div>
            <div class="metric-label">Messages in Session</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        if st.session_state.current_url:
            st.markdown('<div class="section-label">Active Video</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="video-info">
                <div>
                    <div class="video-label">YouTube URL</div>
                    <div class="video-url">{st.session_state.current_url[:55]}...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Powered By</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:#5A6A8A; line-height:2.2;">
         &nbsp;Groq LLaMA 3.3 70B<br>
         &nbsp;FAISS Vector Search<br>
         &nbsp;MiniLM Embeddings<br>
         &nbsp;LangChain Text Splitter
    </div>
    """, unsafe_allow_html=True)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["  Process Video", "  Chat with Video"])

# ---- TAB 1 ----
with tab1:
    st.markdown('<div class="section-label">YouTube Video URL</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        label="url",
        label_visibility="collapsed",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if st.button("  Transcribe & Process Video"):
        if url_input:
            # New URL → clear old chat history
            if url_input != st.session_state.current_url:
                st.session_state.messages = []

            with st.spinner("⏳ Fetching transcript from YouTube..."):
                transcript = get_transcript(url_input)

            if transcript.startswith("ERROR"):
                st.error(f"❌ {transcript}")
            else:
                with st.spinner("⚙️ Building vector embeddings..."):
                    process_transcript(transcript)
                    st.session_state.transcript = transcript
                    st.session_state.current_url = url_input
                    st.session_state.word_count = len(transcript.split())

                st.success("✅ Video processed successfully! Switch to the Chat tab.")

                col1, col2, col3 = st.columns(3)
                col1.metric("Words", f"{st.session_state.word_count:,}")
                col2.metric("Chunks", st.session_state.chunk_count)
                col3.metric("Model", "LLaMA 3.3")

                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                with st.expander("📄 View Full Transcript"):
                    st.markdown(f'<div class="transcript-box">{transcript}</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a YouTube URL to get started.")

    if st.session_state.vector_store is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3rem 2rem; margin-top:1.5rem;">
            <div style="font-size:3rem; margin-bottom:1rem;"></div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#E8EAF0; margin-bottom:0.5rem;">
                No Video Loaded Yet
            </div>
            <div style="font-size:0.88rem; color:#5A6A8A; max-width:320px; margin:0 auto; line-height:1.7;">
                Paste any YouTube URL above and click Transcribe to start analyzing video content with AI.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---- TAB 2 ----
with tab2:
    if st.session_state.vector_store is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 3rem 2rem;">
            <div style="font-size:3rem; margin-bottom:1rem;"></div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#E8EAF0; margin-bottom:0.5rem;">
                Chat Not Available
            </div>
            <div style="font-size:0.88rem; color:#5A6A8A; max-width:300px; margin:0 auto; line-height:1.6;">
                Please process a YouTube video first before starting a conversation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_title, col_clear = st.columns([5, 1])
        with col_title:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
                <span style="font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; color:#E8EAF0;">Chat Session</span>
                <span class="status-badge status-ready">● Active</span>
            </div>
            """, unsafe_allow_html=True)
        with col_clear:
            st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
            if st.button("🗑️ Clear"):
                st.session_state.messages = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:2rem;">
                <div style="font-size:0.9rem; color:#5A6A8A; line-height:1.9;">
                    💡 Ask anything about the video.<br>
                    <span style="font-size:0.82rem; color:#3A4A6A;">
                        Try: "What is the main topic?" &nbsp;·&nbsp; "Summarize key points" &nbsp;·&nbsp; "What did they say about X?"
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

        if question := st.chat_input("Ask a question about the video..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            with st.spinner("Thinking..."):
                answer = generate_answer(question)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
            st.rerun()