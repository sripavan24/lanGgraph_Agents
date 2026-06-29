from langchain_core.messages import HumanMessage
from utils.helpers import extract_name_from_text, extract_email, extract_phone, get_cache_dir

def router_agent(state):
    query = state["messages"][-1].content.lower()

    if any(word in query for word in ["email", "mail", "name", "phone", "mobile", "contact"]):
        return {"next_agent": "general_qa_agent"}

    if any(word in query for word in ["suitable", "sutible", "sutabuil", "fit", "match", "ats", "score", "jd"]):
        return {"next_agent": "ats_score_agent"}

    if any(word in query for word in ["job", "jobs", "opening", "openings", "vacancy", "vacancies", "hiring", "link", "links"]):
        return {"next_agent": "web_search"}

    if any(word in query for word in ["platfo", "website", "websites", "resume maker", "resume builder", "cv maker", "cv builder"]):
        return {"next_agent": "resume_platform_agent"}

    if any(word in query for word in ["suggest", "improve", "update", "section", "change", "fix"]):
        return {"next_agent": "suggestion_agent"}

    prompt = f"""You are a router. Classify the user query and reply with ONLY one word:

Query: {query}

Options: ats_score_agent, suggestion_agent, general_qa_agent, web_search, resume_platform_agent

Use general_qa_agent for questions asking resume facts like name, email, phone, education, skills, projects, or experience.
Use web_search only for job links/openings/hiring questions.
Use resume_platform_agent only for resume builder/platform/website questions.
Use suggestion_agent only when the user asks to improve/change/fix resume sections.
Use ats_score_agent only for ATS score, JD match, suitability, or fit questions."""

    response = state["llm"].invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()
    
    if "web" in decision or "job" in decision:
        return {"next_agent": "web_search"}
    elif "platform" in decision:
        return {"next_agent": "resume_platform_agent"}
    elif "ats" in decision or "score" in decision:
        return {"next_agent": "ats_score_agent"}
    elif "suggest" in decision or "improve" in decision:
        return {"next_agent": "suggestion_agent"}
    else:
        return {"next_agent": "general_qa_agent"}
