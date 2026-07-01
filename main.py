import re
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.ats_score_agent import ats_score_agent
from agents.career_agents import career_agent, github_agent, grammar_agent, interview_agent, linkedin_agent
from agents.general_qa_agent import general_qa_agent
from agents.parser_agent import parser_agent
from agents.router_agent import router_agent
from agents.suggestion_agent import suggestion_agent
from config import embeddings, llm
from tools.web_job_search import web_job_search, web_resume_builder_search


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
    route: str
    profile_analysis: dict
    llm: object
    embeddings: object


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
            return title[:80]
    return " ".join(re.findall(r"\b[A-Za-z+#.]{3,}\b", jd or resume)[:5]) or "software developer"


def job_search_agent(state: AgentState) -> dict:
    return {
        "general_answer": {
            "answer": web_job_search.invoke({
                "role": infer_job_query(state),
                "resume": state.get("raw_text", "")[:5000],
                "job_description": state.get("job_description", "")[:3000],
            })
        }
    }


def resume_platform_agent(state: AgentState) -> dict:
    query = state["messages"][-1].content if state.get("messages") else "ATS resume builder"
    live_results = web_resume_builder_search.invoke({"query": query})
    if not state.get("llm"):
        return {
            "general_answer": {
                "answer": (
                    "I found resume-builder search results, but I need a GROQ_API_KEY or "
                    "GROK_API_KEY value in your .env file to summarize them with the AI model.\n\n"
                    f"{live_results}"
                )
            }
        }
    prompt = f"""Recommend resume builders using these live search results.
Include free, freemium, and paid options with features, ATS compatibility, pricing model, best use case, and official links.

Live results:
{live_results}"""
    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"general_answer": {"answer": response.content}}


def format_answer(response: AgentState) -> str:
    next_agent = response.get("next_agent")
    if next_agent == "ats_score_agent" and response.get("ats_score"):
        score = response["ats_score"]
        suitable = "Yes" if score.get("suitable") is True else "No" if score.get("suitable") is False else "Not sure"
        text = (
            f"ATS Score: {score.get('ats_score', 'N/A')}\n"
            f"Suitable: {suitable}\n\n"
            f"{score.get('summary', '')}"
        ).strip()
        fields = [
            ("Suitable job roles", score.get("suitable_roles")),
            ("Strong matching skills", score.get("strong_skills")),
            ("Missing / weak skills", score.get("missing_skills")),
            ("Reasons", score.get("reasons")),
            ("Improve next", score.get("improvements")),
        ]
        for label, values in fields:
            if values:
                text += f"\n\n{label}:\n" + "\n".join(f"- {value}" for value in values)
        return text
    if response.get("general_answer", {}).get("answer"):
        return response["general_answer"]["answer"]
    if response.get("suggestions", {}).get("details"):
        return response["suggestions"]["details"]
    return "I could not generate an answer for that question."


def build_graph():
    workflow = StateGraph(AgentState)
    nodes = {
        "parser": parser_agent,
        "router": router_agent,
        "ats_score_agent": ats_score_agent,
        "suggestion_agent": suggestion_agent,
        "general_qa_agent": general_qa_agent,
        "web_search": job_search_agent,
        "resume_platform_agent": resume_platform_agent,
        "grammar_agent": grammar_agent,
        "github_agent": github_agent,
        "linkedin_agent": linkedin_agent,
        "interview_agent": interview_agent,
        "career_agent": career_agent,
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.add_edge(START, "parser")
    workflow.add_edge("parser", "router")
    workflow.add_conditional_edges("router", lambda state: state["next_agent"])
    for name in nodes:
        if name not in {"parser", "router"}:
            workflow.add_edge(name, END)
    return workflow.compile()


def create_initial_state(resume_path: str, jd: str, first_message: str = "Analyze ATS score") -> AgentState:
    return {
        "resume_path": resume_path,
        "job_description": jd,
        "raw_text": "",
        "vectorstore": None,
        "messages": [HumanMessage(content=first_message)],
        "llm": llm,
        "embeddings": embeddings,
    }


def run_question(graph, state: AgentState, question: str) -> tuple[AgentState, str]:
    for key in ["suggestions", "general_answer", "next_agent", "route"]:
        state.pop(key, None)
    state["messages"] = [HumanMessage(content=question)]
    response = graph.invoke(state)
    state.update(response)
    return state, format_answer(response)


def main():
    print("AI Resume Analyzer Started")
    resume_path = input("Enter resume PDF path: ")
    jd = input("Paste Job Description: ")
    graph = build_graph()
    state = create_initial_state(resume_path, jd, "ATS score")
    state.update(graph.invoke(state))
    print("Resume loaded.\n")

    while True:
        question = input("Your Question: ")
        if question.lower() in {"exit", "quit", "q"}:
            break
        state, answer = run_question(graph, state, question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
