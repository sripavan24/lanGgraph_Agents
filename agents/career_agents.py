from langchain_core.messages import HumanMessage

from tools.profile_analyzer import analyze_github_profile, analyze_linkedin_profile
from utils.resume_analysis import extract_urls


def llm_agent(state, role: str, instructions: str) -> dict:
    query = state["messages"][-1].content
    if not state.get("llm"):
        return {
            "general_answer": {
                "answer": "I need a GROQ_API_KEY or GROK_API_KEY value in your .env file to answer this with the AI model."
            }
        }

    prompt = f"""{role}

{instructions}

Resume:
{state.get('raw_text', '')[:7000]}

Job description:
{state.get('job_description', '')[:3500]}

User question:
{query}"""
    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"general_answer": {"answer": response.content}}


def grammar_agent(state):
    return llm_agent(
        state,
        "You are a grammar and readability agent.",
        "Review grammar, spelling, readability, tone, and bullets. Provide corrected examples where useful.",
    )


def interview_agent(state):
    return llm_agent(
        state,
        "You are an interview preparation agent.",
        "Create role-specific interview preparation, likely questions, STAR answer guidance, and practice tasks.",
    )


def career_agent(state):
    return llm_agent(
        state,
        "You are a career guidance agent.",
        "Give practical career direction, learning roadmap, positioning, job strategy, and next steps.",
    )


def github_agent(state):
    resume = state.get("raw_text", "")
    urls = extract_urls(resume)
    analysis = (
        analyze_github_profile.invoke({
            "github_url": urls["github"][0],
            "resume": resume[:7000],
            "job_description": state.get("job_description", "")[:3500],
        })
        if urls["github"]
        else {"score": 0, "summary": "No GitHub link was found in the resume.", "evidence": []}
    )
    prompt = f"""You are a GitHub portfolio agent.
Explain whether this GitHub profile strengthens the candidate for the target role.

Analysis:
{analysis}

Resume:
{resume[:5000]}

Job description:
{state.get('job_description', '')[:3000]}"""
    if not state.get("llm"):
        return {"general_answer": {"answer": analysis["summary"]}, "profile_analysis": {"github": analysis}}
    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"general_answer": {"answer": response.content}, "profile_analysis": {"github": analysis}}


def linkedin_agent(state):
    resume = state.get("raw_text", "")
    urls = extract_urls(resume)
    analysis = (
        analyze_linkedin_profile.invoke({
            "linkedin_url": urls["linkedin"][0],
            "resume": resume[:7000],
            "job_description": state.get("job_description", "")[:3500],
        })
        if urls["linkedin"]
        else {"score": 0, "summary": "No LinkedIn link was found in the resume.", "evidence": []}
    )
    prompt = f"""You are a LinkedIn profile agent.
Explain whether this LinkedIn profile strengthens the candidate for the target role.

Analysis:
{analysis}

Resume:
{resume[:5000]}

Job description:
{state.get('job_description', '')[:3000]}"""
    if not state.get("llm"):
        return {"general_answer": {"answer": analysis["summary"]}, "profile_analysis": {"linkedin": analysis}}
    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"general_answer": {"answer": response.content}, "profile_analysis": {"linkedin": analysis}}
