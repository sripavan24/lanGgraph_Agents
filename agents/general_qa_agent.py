from langchain_core.messages import HumanMessage
from utils.helpers import extract_email, extract_jd_designation, extract_name_from_text, extract_phone


def _answer_general_chat(query: str) -> str | None:
    query_lower = " ".join(query.lower().split()).strip(" ?!.")
    if query_lower in {"hi", "hello", "hey", "hii", "hai"}:
        return "Hi there. How are you?"
    if query_lower in {"how are you", "how r u"}:
        return "I'm good, thanks for asking. How can I help you today?"
    if query_lower in {"thanks", "thank you"}:
        return "You're welcome."
    return None


def _extract_section(text: str, section_name: str) -> str:
    lines = text.splitlines()
    start = None
    section_headers = (
        "summary", "profile", "objective", "education", "experience", "work experience",
        "skills", "technical skills", "projects", "certifications", "achievements",
        "contact", "languages", "interests",
    )
    for index, line in enumerate(lines):
        normalized = line.strip().lower().strip(":")
        if normalized == section_name:
            start = index + 1
            break
    if start is None:
        return ""

    collected = []
    for line in lines[start:]:
        normalized = line.strip().lower().strip(":")
        if collected and normalized in section_headers:
            break
        if line.strip():
            collected.append(line.strip())
    return "\n".join(collected[:10])


def _answer_resume_fact(query: str, raw_text: str, jd: str = "") -> str | None:
    query_lower = query.lower()
    if "jd" in query_lower and any(word in query_lower for word in ["designation", "title", "role", "position", "name"]):
        return f"The JD designation is {extract_jd_designation(jd)}."
    if "name" in query_lower:
        return f"The candidate name is {extract_name_from_text(raw_text)}."
    if "email" in query_lower or "mail" in query_lower:
        return f"The email is {extract_email(raw_text)}."
    if "phone" in query_lower or "mobile" in query_lower or "contact" in query_lower:
        return f"The phone number is {extract_phone(raw_text)}."
    if "education" in query_lower or "college" in query_lower or "degree" in query_lower or "university" in query_lower:
        education = _extract_section(raw_text, "education")
        return f"Education:\n{education}" if education else "I could not find a clear education section in the resume."
    return None

def general_qa_agent(state):
    query = state["messages"][-1].content
    raw_text = state.get("raw_text", "")
    jd = state.get("job_description", "")
    if state.get("route") == "general_chat":
        direct_chat = _answer_general_chat(query)
        if direct_chat:
            return {"general_answer": {"answer": direct_chat}}

    direct_answer = _answer_resume_fact(query, raw_text, jd)
    if direct_answer:
        return {"general_answer": {"answer": direct_answer}}

    if not state.get("llm"):
        if state.get("route") == "general_chat":
            return {
                "general_answer": {
                    "answer": "I can answer greetings, but open-ended AI chat needs a GROQ_API_KEY or GROK_API_KEY value in your .env file."
                }
            }
        return {
            "general_answer": {
                "answer": "I need a GROQ_API_KEY or GROK_API_KEY value in your .env file to answer this with the AI model."
            }
        }

    if state.get("route") == "general_chat":
        prompt = f"""You are a friendly general assistant.
Answer the user's message naturally and briefly.
Do not give resume advice unless the user asks about resume, ATS, jobs, interviews, or career topics.

User message: {query}"""
    else:
        prompt = f"""You are a helpful resume question-answering assistant.
Answer the user's exact question first. If the user asks for a resume fact, extract it from the resume and do not turn the answer into resume-improvement advice.
Use the job description only when the question asks about fit, ATS, matching, or career guidance.
If the answer is not present in the resume or job description, say that clearly.

Resume:
{raw_text[:8000]}

Job Description:
{jd[:4000]}

Question: {query}"""

    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"general_answer": {"answer": response.content}}
