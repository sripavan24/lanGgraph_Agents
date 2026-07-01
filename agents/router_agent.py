from langchain_core.messages import HumanMessage

AGENT_OPTIONS = {
    "resume_analysis": "Deep resume review, resume facts, section extraction, or resume quality questions.",
    "ats": "ATS score, JD match, suitability, parsing, keyword fit, and compatibility.",
    "grammar": "Grammar, spelling, clarity, readability, and bullet rewriting.",
    "github": "GitHub profile, repositories, technical portfolio, and code presence.",
    "linkedin": "LinkedIn profile, professional brand, and profile completeness.",
    "job_search": "Live job recommendations, application links, company roles, internships, and openings.",
    "resume_suggestions": "Concrete resume improvements, bullet rewrites, missing sections, and optimization plan.",
    "resume_platforms": "Resume builders, CV builders, templates, ATS-friendly resume platforms, and pricing.",
    "interview": "Interview preparation, mock questions, STAR answers, and role-specific practice.",
    "career": "Career guidance, roadmap, role selection, learning plan, and salary or growth advice.",
    "general_chat": "General career conversation or anything not covered by the specialist agents.",
}

GRAPH_AGENT_MAP = {
    "resume_analysis": "general_qa_agent",
    "ats": "ats_score_agent",
    "grammar": "grammar_agent",
    "github": "github_agent",
    "linkedin": "linkedin_agent",
    "job_search": "web_search",
    "resume_suggestions": "suggestion_agent",
    "resume_platforms": "resume_platform_agent",
    "interview": "interview_agent",
    "career": "career_agent",
    "general_chat": "general_qa_agent",
}

RESUME_FACT_PATTERNS = (
    "name",
    "email",
    "mail",
    "phone",
    "mobile",
    "contact",
    "github",
    "linkedin",
    "education",
    "college",
    "degree",
    "university",
)

GENERAL_CHAT_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hii",
    "hai",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "how r u",
    "thanks",
    "thank you",
}

RESUME_INTENT_WORDS = (
    "resume",
    "cv",
    "ats",
    "score",
    "improve",
    "improvement",
    "suggest",
    "fix",
    "rewrite",
    "job description",
    "jd",
    "match",
    "suitable",
    "skills",
    "project",
    "experience",
    "interview",
    "career",
    "job",
    "jobs",
    "github",
    "linkedin",
)


def _keyword_route(query_lower: str) -> tuple[str, str]:
    if any(word in query_lower for word in ["job", "jobs", "opening", "openings", "vacancy", "hiring"]):
        return "web_search", "job_search"
    if any(word in query_lower for word in ["resume builder", "cv builder", "resume maker", "template", "platform"]):
        return "resume_platform_agent", "resume_platforms"
    if any(word in query_lower for word in ["ats", "score", "match", "suitable", "fit", "jd", "job description"]):
        return "ats_score_agent", "ats"
    if any(word in query_lower for word in ["grammar", "spelling", "readability", "rewrite"]):
        return "grammar_agent", "grammar"
    if any(word in query_lower for word in ["interview", "mock", "questions"]):
        return "interview_agent", "interview"
    if any(word in query_lower for word in ["career", "roadmap", "learning", "salary"]):
        return "career_agent", "career"
    if any(word in query_lower for word in ["improve", "improvement", "suggest", "fix", "resume", "cv"]):
        return "suggestion_agent", "resume_suggestions"
    return "general_qa_agent", "general_chat"


def router_agent(state):
    query = state["messages"][-1].content
    query_lower = query.lower()
    compact_query = " ".join(query_lower.split()).strip(" ?!.")
    if compact_query in GENERAL_CHAT_PHRASES:
        return {"next_agent": "general_qa_agent", "route": "general_chat"}
    if any(pattern in query_lower for pattern in RESUME_FACT_PATTERNS):
        return {"next_agent": "general_qa_agent", "route": "resume_analysis"}
    if not any(word in query_lower for word in RESUME_INTENT_WORDS):
        return {"next_agent": "general_qa_agent", "route": "general_chat"}
    if not state.get("llm"):
        next_agent, route = _keyword_route(query_lower)
        return {"next_agent": next_agent, "route": route}

    options = "\n".join(f"- {name}: {description}" for name, description in AGENT_OPTIONS.items())
    prompt = f"""You are an LLM Router Agent for an AI career assistant.
Choose the single best specialist agent by semantic intent, not by keyword matching.
Return only the agent id.

Available agents:
{options}

User query:
{query}

Resume context exists: {bool(state.get('raw_text'))}
Job description exists: {bool(state.get('job_description'))}"""
    response = state["llm"].invoke([HumanMessage(content=prompt)])
    agent_id = response.content.strip().lower().split()[0].strip("`.,:")
    return {"next_agent": GRAPH_AGENT_MAP.get(agent_id, "general_qa_agent"), "route": agent_id}
