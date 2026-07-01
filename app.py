from pathlib import Path
from html import escape

import markdown as md
import streamlit as st
from langchain_core.messages import HumanMessage

from agents.general_qa_agent import _answer_general_chat
from config import llm
from main import build_graph, create_initial_state, run_question


st.set_page_config(page_title="AI Resume Analyzer", layout="wide")


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def app_css() -> str:
    return """
<style>
:root {
  --bg: #eef6ff;
  --panel: rgba(255, 255, 255, .66);
  --panel-strong: rgba(255, 255, 255, .9);
  --text: #102033;
  --muted: #64748b;
  --line: rgba(87, 111, 141, .18);
  --accent: #0ea5e9;
  --shadow: 0 24px 70px rgba(30, 64, 175, .14);
}
@property --progress {
  syntax: '<number>';
  inherits: true;
  initial-value: 0;
}
.stApp {
  background:
    linear-gradient(135deg, rgba(219, 234, 254, .95), rgba(248, 250, 252, .96) 48%, rgba(224, 242, 254, .92)),
    radial-gradient(circle at 14% 12%, rgba(14, 165, 233, .2), transparent 32%),
    radial-gradient(circle at 88% 6%, rgba(34, 197, 94, .12), transparent 28%);
  color: var(--text);
  overflow: auto;
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
  max-width: none;
  min-height: 100vh;
  padding: 14px 16px 22px;
  overflow: visible;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border-radius: 24px;
  overflow: visible;
}
div[data-testid="column"]:first-of-type div[data-testid="stVerticalBlockBorderWrapper"] {
  min-height: calc(100vh - 36px);
  padding: 18px;
  overflow: visible;
  overflow-x: hidden;
}
div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] {
  min-height: calc(100vh - 36px);
  padding: 0;
  overflow: visible;
}
div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlock"] {
  min-height: calc(100vh - 36px);
}
div[data-testid="column"]:first-of-type div[data-testid="stVerticalBlock"] {
  gap: .52rem;
}
label, [data-testid="stWidgetLabel"] p, .stMarkdown, .stMarkdown p {
  color: var(--text) !important;
}
[data-testid="stWidgetLabel"] p {
  font-weight: 750 !important;
}
.brand {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 12px 0 10px;
}
.brand h1 { font-size: 1.05rem; margin: 0; letter-spacing: 0; color: var(--text); }
.score-card {
  background: linear-gradient(180deg, rgba(255,255,255,.76), rgba(255,255,255,.48));
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 14px;
  margin: 10px 0;
}
.score-card h3 {
  margin: 0;
  color: var(--text);
  font-size: .9rem;
}
.score-card .score-out {
  margin: 3px 0 0;
  color: var(--muted);
  font-size: .82rem;
}
.score-wrap { display: grid; place-items: center; padding: 8px 0 4px; }
.score-ring {
  --progress: 0;
  width: min(152px, 62vw);
  aspect-ratio: 1;
  border-radius: 50%;
  display: grid;
  place-items: center;
  position: relative;
  background: conic-gradient(var(--accent) calc(var(--progress) * 1%), rgba(148,163,184,.2) 0);
  animation: fillScore 1.15s ease-out forwards;
}
.score-ring::after {
  content: "";
  position: absolute;
  inset: 12px;
  border-radius: 50%;
  background: var(--panel-strong);
  border: 1px solid var(--line);
}
.score-value { position: relative; z-index: 1; text-align: center; }
.score-value strong { display: block; font-size: 2.7rem; line-height: .9; color: var(--text); }
.score-value span { display: block; color: var(--muted); font-size: .9rem; font-weight: 700; margin-top: 3px; }
.score-value small { display: block; color: var(--muted); font-size: .78rem; margin-top: 4px; }
@keyframes fillScore {
  from { --progress: 0; }
  to { --progress: var(--score); }
}
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 14px; }
.pill {
  border: 1px solid rgba(14,165,233,.2);
  background: rgba(255,255,255,.58);
  border-radius: 999px;
  padding: 7px 10px;
  font-size: .78rem;
  color: #1e5f8d;
  font-weight: 650;
}
.chat-scroll {
  min-height: calc(100vh - 126px);
  max-height: calc(100vh - 126px);
  overflow-y: auto;
  padding: 18px 22px;
}
.chat-row {
  display: flex;
  width: 100%;
  margin: 0 0 12px;
}
.chat-row.user { justify-content: flex-end; }
.chat-row.assistant { justify-content: flex-start; }
.chat-bubble {
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 12px 14px;
  line-height: 1.52;
  color: var(--text);
  max-width: 78%;
  background: rgba(255,255,255,.62);
  overflow-wrap: anywhere;
}
.chat-row.user .chat-bubble {
  background: linear-gradient(135deg, #0ea5e9, #2563eb);
  color: white;
  border-color: rgba(255,255,255,.3);
}
.chat-row.user .chat-bubble p,
.chat-row.user .chat-bubble li {
  color: white !important;
}
.chat-bubble p:first-child { margin-top: 0; }
.chat-bubble p:last-child { margin-bottom: 0; }
.chat-empty {
  min-height: calc(100vh - 170px);
  display: grid;
  place-items: center;
  color: var(--muted);
  text-align: center;
}
.chat-form {
  border-top: 1px solid rgba(87,111,141,.12);
  padding: 12px 16px 16px;
  background: rgba(255,255,255,.42);
}
div[data-testid="stForm"] {
  border: 0;
  padding: 0;
  background: transparent;
}
div[data-testid="stForm"] [data-testid="stTextInput"] input {
  border-radius: 18px;
  min-height: 48px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,.78);
}
div[data-testid="stForm"] button {
  min-height: 48px;
  border-radius: 16px;
  background: #0ea5e9;
  border: 0;
  color: white;
  font-weight: 800;
}
div[data-testid="stButton"] button[kind="primary"] {
  background: #0ea5e9;
  border: 0;
  color: white;
  border-radius: 14px;
  min-height: 44px;
  font-weight: 800;
}
[data-testid="stFileUploader"] section,
[data-testid="stTextArea"] textarea {
  border-radius: 16px;
}
@media (max-width: 920px) {
  .block-container { height: auto; overflow: visible; padding: 10px; }
  div[data-testid="column"]:first-of-type div[data-testid="stVerticalBlockBorderWrapper"],
  div[data-testid="column"]:nth-of-type(2) div[data-testid="stVerticalBlockBorderWrapper"] {
    height: auto;
    min-height: 70vh;
  }
  .chat-scroll { max-height: none; min-height: 520px; }
  .chat-empty { min-height: 460px; }
}
</style>
"""


@st.cache_resource
def graph_resource():
    return build_graph()


def save_uploaded_file(uploaded_file) -> str:
    upload_dir = Path("resume_cache/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return str(path)


def render_score(score: int):
    html(
        f"""
        <div class="score-card">
          <h3>ATS Score</h3>
          <p class="score-out">{score} / 100</p>
        <div class="score-wrap">
          <div class="score-ring" style="--score:{score}">
            <div class="score-value">
              <strong>{score}</strong>
              <span>/100</span>
              <small>ATS Score</small>
            </div>
          </div>
        </div>
        </div>
        """
    )


def render_dashboard(analysis: dict | None):
    score = int((analysis or {}).get("ats_score", 0))
    strong = (analysis or {}).get("strong_skills", [])[:5]
    preferred_skills = {
        "python", "django", "machine learning", "ai", "artificial intelligence",
        "sql", "fastapi", "flask", "react", "javascript", "typescript", "java",
        "aws", "docker", "langchain", "llm", "data analysis", "pandas", "numpy",
    }
    important_skills = []
    for skill in strong:
        normalized = str(skill).lower()
        if normalized in preferred_skills or any(term in normalized for term in preferred_skills):
            important_skills.append(skill)
    important_skills = important_skills[:7]

    html('<div class="brand"><div><h1>AI Resume Analyzer</h1></div></div>')
    render_score(score)
    html(
        '<div class="pill-row">'
        + "".join(f'<span class="pill">{escape(str(skill))}</span>' for skill in important_skills)
        + "</div>"
    )


def messages_markup(history: list[dict]) -> str:
    if not history:
        return '<div class="chat-empty">Ask anything.</div>'
    markup = ""
    for message in history:
        role = "user" if message.get("role") == "user" else "assistant"
        content = md.markdown(escape(message.get("content", "")), extensions=["sane_lists"])
        markup += f'<div class="chat-row {role}"><div class="chat-bubble">{content}</div></div>'
    return markup


def render_messages(history: list[dict]):
    html(f'<div class="chat-scroll">{messages_markup(history)}</div>')


def answer_without_resume(prompt: str) -> str:
    direct = _answer_general_chat(prompt)
    if direct:
        return direct
    resume_terms = (
        "resume", "cv", "ats", "score", "job description", "jd", "improve",
        "suggest", "fix", "rewrite", "match", "suitable", "education",
        "experience", "skills", "project", "github", "linkedin",
    )
    if any(term in prompt.lower() for term in resume_terms):
        return "Please analyze a resume first so I can answer that properly."
    if llm is None:
        return "I can answer greetings, but open-ended AI chat needs a GROQ_API_KEY or GROK_API_KEY value in your .env file."
    response = llm.invoke([HumanMessage(content=f"Answer briefly and naturally.\n\nUser: {prompt}")])
    return response.content


if "history" not in st.session_state:
    st.session_state.history = []
if "state" not in st.session_state:
    st.session_state.state = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None

html(app_css())

left, right = st.columns([0.32, 0.68], gap="medium")

with left:
    with st.container(border=True):
        uploaded = st.file_uploader("Resume PDF", type=["pdf"])
        jd = st.text_area("Job description", height=170, placeholder="Paste the target job description here...")
        analyze = st.button("Analyze resume", type="primary", use_container_width=True)

        if analyze and uploaded:
            with st.spinner("Parsing resume, scoring ATS, and checking profiles..."):
                resume_path = save_uploaded_file(uploaded)
                graph = graph_resource()
                state = create_initial_state(resume_path, jd, "Analyze ATS score")
                state.update(graph.invoke(state))
                st.session_state.state = state
                st.session_state.analysis = state.get("ats_score")
                st.session_state.history = [
                    {"role": "assistant", "content": "Resume analyzed. Ask me any question about it."}
                ]
        elif analyze:
            st.warning("Upload a PDF resume first.")

        render_dashboard(st.session_state.analysis)

with right:
    with st.container(border=True):
        render_messages(st.session_state.history)

        with st.form("career_chat_form", clear_on_submit=True):
            input_col, send_col = st.columns([0.86, 0.14], gap="small")
            with input_col:
                prompt = st.text_input(
                    "Chat message",
                    placeholder="Ask a question about the resume...",
                    label_visibility="collapsed",
                )
            with send_col:
                submitted = st.form_submit_button("Send", use_container_width=True)

        if not submitted:
            prompt = ""

        if prompt:
            st.session_state.history.append({"role": "user", "content": prompt})
            if not st.session_state.state:
                with st.spinner("Thinking..."):
                    answer = answer_without_resume(prompt)
                st.session_state.history.append({"role": "assistant", "content": answer})
                st.rerun()
            else:
                with st.spinner("Thinking..."):
                    graph = graph_resource()
                    state, answer = run_question(graph, st.session_state.state, prompt)
                    st.session_state.state = state
                    if state.get("ats_score"):
                        st.session_state.analysis = state["ats_score"]
                st.session_state.history.append({"role": "assistant", "content": answer})
                st.rerun()
