from langchain_core.messages import HumanMessage
import json
import re
from tools.profile_analyzer import analyze_github_profile, analyze_linkedin_profile
from utils.resume_analysis import build_resume_analysis, extract_urls, safe_json

def ats_score_agent(state):
    resume = state.get("raw_text", "")
    jd = state.get("job_description", "")
    query = state["messages"][-1].content if state.get("messages") else ""
    profile_analysis = state.get("profile_analysis")

    if profile_analysis is None:
        urls = extract_urls(resume)
        profile_analysis = {}
        if urls["github"]:
            profile_analysis["github"] = analyze_github_profile.invoke({
                "github_url": urls["github"][0],
                "resume": resume[:6000],
                "job_description": jd[:3000],
            })
        if urls["linkedin"]:
            profile_analysis["linkedin"] = analyze_linkedin_profile.invoke({
                "linkedin_url": urls["linkedin"][0],
                "resume": resume[:6000],
                "job_description": jd[:3000],
            })

    deterministic = build_resume_analysis(resume, jd, profile_analysis)
    if not state.get("llm"):
        deterministic["profile_analysis"] = profile_analysis
        return {"ats_score": deterministic, "profile_analysis": profile_analysis}

    prompt = f"""You are an ATS resume analysis agent.
The deterministic scoring engine has already analyzed the resume. Do not invent a new score.
Add concise human-readable context while preserving the JSON schema and score.

User Question:
{query}

Deterministic analysis JSON:
{json.dumps(deterministic, indent=2)}

Resume excerpt:
{resume[:6000]}

Return ONLY valid JSON with these keys:
ats_score: number 0-100,
suitable: true or false,
summary: short direct answer,
categories: object,
suitable_roles: array of matching job roles,
strong_skills: array,
missing_skills: array,
reasons: array,
improvements: array."""

    response = state["llm"].invoke([HumanMessage(content=prompt)])
    score_data = safe_json(response.content, deterministic)
    score_data["ats_score"] = deterministic["ats_score"]
    score_data["categories"] = deterministic["categories"]
    score_data["profile_analysis"] = profile_analysis
    score_data["section_scores"] = deterministic["section_scores"]
    score_data["profile_links"] = deterministic["profile_links"]

    return {"ats_score": score_data, "profile_analysis": profile_analysis}
