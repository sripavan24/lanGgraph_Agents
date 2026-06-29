from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
import re
from config import llm, embeddings
from agents.parser_agent import parser_agent
from agents.router_agent import router_agent
from agents.ats_score_agent import ats_score_agent
from agents.suggestion_agent import suggestion_agent
from agents.general_qa_agent import general_qa_agent
from tools.web_job_search import web_job_search

RESUME_PLATFORMS = """Here are good resume update/building platforms:

1. Canva Resume Builder - https://www.canva.com/resumes/
2. Novoresume - https://novoresume.com/
3. Resume.io - https://resume.io/
4. Zety - https://zety.com/resume-builder
5. Enhancv - https://enhancv.com/
6. Teal Resume Builder - https://www.tealhq.com/resume-builder
7. FlowCV - https://flowcv.com/

For ATS-friendly resumes, prefer simple layouts, clear section names, measurable project points, and PDF export."""

class AgentState(TypedDict, total=False):
    resume_path: str
    job_description: str
    raw_text: str
    vectorstore: object
    ats_score: dict
    suggestions: dict
    general_answer: dict
    messages: list
    next_agent: str
    llm: object
    embeddings: object

def get_ats_score(state: AgentState) -> int:
    score = state.get("ats_score", {}).get("ats_score", 0)
    try:
        return int(score)
    except (TypeError, ValueError):
        return 0

def infer_job_query(state: AgentState) -> str:
    jd = state.get("job_description", "")
    resume = state.get("raw_text", "")
    text = f"{jd}\n{resume}"

    title_patterns = [
        r"(?i)(?:job title|role|position)\s*[:\-]\s*([A-Za-z0-9 .,+#/-]{3,80})",
        r"(?i)\b(hiring|looking for|opening for)\s+(?:an?|the)?\s*([A-Za-z0-9 .,+#/-]{3,80})",
        r"(?i)\b(?:software|frontend|backend|full stack|data|machine learning|ai|python|java|react|devops|cloud|web)\s+(?:engineer|developer|analyst|intern|architect)\b",
    ]

    for pattern in title_patterns:
        match = re.search(pattern, text)
        if match:
            title = match.group(match.lastindex or 0).strip(" .,-")
            if title.lower() not in {"hiring", "looking for", "opening for"}:
                return title[:80]

    skills = []
    common_skills = [
        "python", "java", "javascript", "typescript", "react", "node", "sql",
        "machine learning", "data analysis", "django", "flask", "fastapi",
        "aws", "azure", "docker", "kubernetes", "html", "css", "mongodb",
    ]
    lower_text = text.lower()
    for skill in common_skills:
        if skill in lower_text:
            skills.append(skill)
        if len(skills) == 4:
            break

    if skills:
        return " ".join(skills)

    return "resume matching"

def job_search_agent(state: AgentState) -> dict:
    score = get_ats_score(state)
    if score and score < 75:
        return {
            "general_answer": {
                "answer": (
                    f"Your ATS score is {score}, so improve the resume before applying.\n\n"
                    "Main reasons to improve toward 75+:\n"
                    "1. Add more JD keywords naturally in skills, experience, and projects.\n"
                    "2. Match your project bullets to the role responsibilities.\n"
                    "3. Add measurable impact, tools, and technologies used.\n"
                    "4. Keep section names simple: Skills, Experience, Projects, Education.\n\n"
                    "Ask: tell me which section I should improve"
                )
            }
        }

    return {
        "general_answer": {
            "answer": web_job_search.invoke({"role": infer_job_query(state)})
        }
    }

def resume_platform_agent(state: AgentState) -> dict:
    return {"general_answer": {"answer": RESUME_PLATFORMS}}

def format_answer(response: AgentState) -> str:
    next_agent = response.get("next_agent")

    if next_agent in {"web_search", "resume_platform_agent", "general_qa_agent"}:
        return response.get("general_answer", {}).get("answer", "I could not generate an answer for that question.")

    if next_agent == "ats_score_agent" and response.get("ats_score"):
        score = response["ats_score"]
        suitable = score.get("suitable")
        suitable_text = "Yes" if suitable is True else "No" if suitable is False else "Not sure"
        text = (
            f"ATS Score: {score.get('ats_score', 'N/A')}\n"
            f"Suitable: {suitable_text}\n\n"
            f"{score.get('summary', '')}"
        ).strip()
        if score.get("suitable_roles"):
            text += "\n\nSuitable job roles:\n" + "\n".join(f"- {role}" for role in score["suitable_roles"])
        if score.get("strong_skills"):
            text += "\n\nStrong matching skills:\n" + "\n".join(f"- {skill}" for skill in score["strong_skills"])
        if score.get("missing_skills"):
            text += "\n\nMissing / weak skills:\n" + "\n".join(f"- {skill}" for skill in score["missing_skills"])
        if score.get("reasons"):
            text += "\n\nReasons:\n" + "\n".join(f"- {reason}" for reason in score["reasons"])
        if score.get("improvements"):
            text += "\n\nImprove next:\n" + "\n".join(f"- {item}" for item in score["improvements"])
        if response.get("suggestions", {}).get("details"):
            text += f"\n\nSuggestions:\n{response['suggestions']['details']}"
        return text

    if next_agent == "suggestion_agent" and response.get("suggestions", {}).get("details"):
        return response["suggestions"]["details"]

    if response.get("general_answer", {}).get("answer"):
        return response["general_answer"]["answer"]

    if response.get("suggestions", {}).get("details"):
        return response["suggestions"]["details"]

    if response.get("ats_score"):
        return str(response["ats_score"])

    return "I could not generate an answer for that question."

def main():
    print("🚀 AI Resume Analyzer Started")
    
    resume_path = input("Enter resume PDF path: ")
    jd = input("Paste Job Description: ")

    workflow = StateGraph(AgentState)
    workflow.add_node("parser", parser_agent)
    workflow.add_node("router", router_agent)
    workflow.add_node("ats_score_agent", ats_score_agent)
    workflow.add_node("suggestion_agent", suggestion_agent)
    workflow.add_node("general_qa_agent", general_qa_agent)
    workflow.add_node("web_search", job_search_agent)
    workflow.add_node("resume_platform_agent", resume_platform_agent)

    workflow.add_edge(START, "parser")
    workflow.add_edge("parser", "router")
    workflow.add_conditional_edges("router", lambda s: s["next_agent"])
    
    workflow.add_edge("ats_score_agent", END)
    workflow.add_edge("web_search", END)
    workflow.add_edge("resume_platform_agent", END)
    workflow.add_edge("suggestion_agent", END)
    workflow.add_edge("general_qa_agent", END)

    graph = workflow.compile()

    # Initial run
    state = {
        "resume_path": resume_path,
        "job_description": jd,
        "raw_text": "",
        "vectorstore": None,
        "messages": [HumanMessage(content="ats score")],
        "llm": llm,
        "embeddings": embeddings
    }
    result = graph.invoke(state)
    state.update(result)
    print("✅ Resume Loaded!\n")

    while True:
        q = input("💬 Your Question: ")
        if q.lower() in {"exit", "quit", "q"}:
            break
        for key in ["ats_score", "suggestions", "general_answer", "next_agent"]:
            state.pop(key, None)
        state["messages"] = [HumanMessage(content=q)]
        response = graph.invoke(state)
        state.update(response)
        print("\nAnswer:", format_answer(response))

if __name__ == "__main__":
    main()
